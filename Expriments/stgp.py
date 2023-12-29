import pandas as pd
import random as random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from sklearn.model_selection import train_test_split
from ta.trend import SMAIndicator
import util as util
import sys


def run_gp(stock_name, trend,gp_max_gen):
    util.DEAP_Monkey_Patching()
    if __name__ != "__main__":
        util.read_candles()
    util.seed(16)
    adf = util.read_candles_one(stock_name)
    adf['gross_return'] = adf['adjclose'] / adf['adjclose'].shift(1)
    adf['return'] = adf['adjclose'].pct_change()
    train_df, test_df = train_test_split(adf, test_size=0.2, shuffle=False)
    ma_cache = {}

    def ma(x, n):
        k = (x.index[0], x.index[-1], n)
        if k not in ma_cache:
            ma_cache[k] = SMAIndicator(x, n).sma_indicator()
        return ma_cache[k]

    lag_cache = {}

    def lag(x, n):
        k = (x.index[0], x.index[-1], n)
        if k not in lag_cache:
            lag_cache[k] = x.shift(n)
        return lag_cache[k]

    class RootElem:
        def __init__(self, tree_result, hold_count):
            self.tree_result=tree_result
            self.hold_count=hold_count


    class Const:
        pass

    def root_calculator(tree_result,hold_count):
        return RootElem(tree_result,hold_count)

    pset = gp.PrimitiveSetTyped('MAIN', [float, float, float, float], RootElem)

    pset.renameArguments(ARG0="open")
    pset.renameArguments(ARG1="high")
    pset.renameArguments(ARG2="low")
    pset.renameArguments(ARG3="adjclose")

    pset.addPrimitive(np.add, [float, float], float)
    pset.addPrimitive(np.multiply, [Const, float], float)

    pset.addPrimitive(lag, [float, Const], float)
    pset.addPrimitive(ma, [float, Const], float)
    pset.addPrimitive(root_calculator,[bool,Const],RootElem)

    pset.addPrimitive(np.logical_and, [bool, bool], bool)
    pset.addPrimitive(np.logical_or, [bool, bool], bool)
    pset.addPrimitive(np.logical_not, [bool], bool)

    pset.addPrimitive(np.less, [float, float], bool)
    pset.addEphemeralConstant("rand"+stock_name, lambda: random.randint(1, 100), Const)

    pset.addTerminal(True, bool)
    pset.addTerminal(False, bool)

    def extend(f):
        return f

    pset.addPrimitive(extend, [Const], Const)

    # strategy 1
    # buy at next buy signal and sell after n candles
    # do not buy again before sell
    def buy_hold_n_candles(df, buy, n):
        i = 0
        b = buy.values
        while i < len(b):
            if b[i]:
                b[i:i + n] = True
                i += n
            else:
                i += 1
        bb = buy.shift(1).fillna(False).values
        return df[bb]['gross_return'].prod()

    def eval_rule2(ind, df):
        key = str(ind)
        args = {k: df[k] for k in ['open', 'high', 'low',
                                   'adjclose']}  # need to call func over whole columns because of operators like ma, lag
        func = toolbox.compile(expr=ind)
        result = func(**{**args})
        if not isinstance(result.tree_result, pd.Series):  # some rules like or_(True, False) returns scalar instead of series
            result.tree_result = pd.DataFrame(index=df.index, data=[result.tree_result] * len(df))
        n = result.hold_count
        ret = buy_hold_n_candles(df, result.tree_result, n)
        # ret = buy_sell(df,act)
        return ret,

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", lambda ind: eval_rule2(ind, train_df))
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    def run(i):
        pop = toolbox.population(n=100)
        pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.1, i, stats=stats, halloffame=hof, verbose=True)

    run(gp_max_gen)

    def report(i=0):
        ind = hof[i]
        print(f'rule: {ind}')
        train_fitness = ind.fitness.values
        print(f'train fitness: {train_fitness}')
        test_fitness = eval_rule2(ind, test_df)
        print(f'test fitness: {test_fitness}')
        bh_fitness = test_df['gross_return'].iloc[1:].prod()
        print(f'buy&hold fitness: {bh_fitness}')
        print(f'model - buy&hold = {test_fitness[0] - bh_fitness}')
        ind1 = str(ind).replace(',', '-')
        tmp = f'{trend},{stock_name},{train_fitness[0]},{test_fitness[0]},{bh_fitness},20,{ind1}\n'
        with open(f'./output/outputup-Random-gp-{gp_max_gen}.csv', 'a') as fd:
            fd.write(tmp)

    report()

if __name__ == "__main__":
    stock_name = str(sys.argv[1])
    stock_trend = str(sys.argv[2])
    gp_max_gen=int(sys.argv[3])
    util.read_candles()
    run_gp(stock_name,stock_trend,gp_max_gen)