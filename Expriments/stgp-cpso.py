import pandas as pd
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from sklearn.model_selection import train_test_split
from ta.trend import SMAIndicator
import pyswarms as ps
import util as util
import random
import sys


def run_gp(stock_name, trend, const_multiplier,max_pso_itter):
    util.DEAP_Monkey_Patching()
    util.seed(16)
    adf = util.read_candles_one(stock_name)
    adf['gross_return'] = adf['adjclose'] / adf['adjclose'].shift(1)
    train_df, test_df = train_test_split(adf, test_size=0.2, shuffle=False)

    ma_cache = {}

    def ma(x, n):
        n = int(n * const_multiplier) + 1  # pso gives [0, 1]
        k = (x.index[0], x.index[-1], n)
        if k not in ma_cache:
            ma_cache[k] = SMAIndicator(x, n).sma_indicator()
        return ma_cache[k]

    lag_cache = {}

    def lag(x, n):
        n = int(n * const_multiplier) + 1  # pso gives [0, 1]
        k = (x.index[0], x.index[-1], n)
        if k not in lag_cache:
            lag_cache[k] = x.shift(n)
        return lag_cache[k]

    class Const:
        pass

    pset = gp.PrimitiveSetTyped('MAIN', [float, float, float, float, Const, Const, Const, Const], bool)

    pset.renameArguments(ARG0="open")
    pset.renameArguments(ARG1="high")
    pset.renameArguments(ARG2="low")
    pset.renameArguments(ARG3="adjclose")
    pset.renameArguments(ARG4="c1")
    pset.renameArguments(ARG5="c2")
    pset.renameArguments(ARG6="c3")
    pset.renameArguments(ARG7="c4")
    #pset.renameArguments(ARG8="c5")

    pset.addPrimitive(np.add, [float, float], float)
    pset.addPrimitive(np.multiply, [Const, float], float)
    pset.addPrimitive(lag, [float, Const], float)
    pset.addPrimitive(ma, [float, Const], float)
    pset.addPrimitive(np.logical_and, [bool, bool], bool)
    pset.addPrimitive(np.logical_or, [bool, bool], bool)
    pset.addPrimitive(np.logical_not, [bool], bool)
    pset.addPrimitive(np.less, [float, float], bool)

    pset.addTerminal(True, bool)
    pset.addTerminal(False, bool)

    def extend(f):
        return f

    pset.addPrimitive(extend, [Const], Const)

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

    pso_bounds = ((0, 0, 0,0,0), (1, 1, 1,1,1))
    pso_options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizers = {}

    def eval_rule2(ind, pos, df):  # for pso, lower level, returns -fitness as pso alg can minimizes only
        rets = []
        for r in pos:
            consts = {f'c{i + 1}': r[i] for i in range(len(r) - 1)}
            n = int(r[-1] * const_multiplier) + 1
            # n = 5
            args = {k: df[k] for k in ['open', 'high', 'low',
                                       'adjclose']}  # need to call func over whole columns because of operators like ma, lag
            func = toolbox.compile(expr=ind)
            act = func(**{**args, **consts})
            if not isinstance(act, pd.Series):  # some rules like or_(True, False) returns scalar instead of series
                act = pd.DataFrame(index=df.index, data=[act] * len(df))
            ret = buy_hold_n_candles(df, act, n)
            rets.append(-ret)
        return np.array(rets)

    def eval_rule(ind, df):  # for gp, higher level
        key = str(ind)
        if key not in optimizers:
            optimizers[key] = ps.single.GlobalBestPSO(n_particles=10, dimensions=5, options=pso_options,
                                                      bounds=pso_bounds)
        cost, pos = optimizers[key].optimize(lambda x: eval_rule2(ind, x, df), iters=max_pso_itter, verbose=False)
        optimizers[key]._cost = -cost
        optimizers[key]._pos = pos
        return optimizers[key]._cost,

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", lambda ind: eval_rule(ind, train_df))
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

    run(20)

    def report(i=0):
        ind = hof[i]
        key = str(ind)
        print(f'rule: {ind}')
        train_fitness = optimizers[key]._cost
        consts = optimizers[key]._pos
        print(f'train fitness: {train_fitness}')
        print(f'train consts: {consts}')
        test_fitness = -eval_rule2(ind, np.array([consts]), test_df)
        print(f'test fitness: {test_fitness}')
        bh_fitness = test_df['gross_return'].iloc[1:].prod()
        print(f'buy&hold fitness: {bh_fitness}')
        print(f'buy&hold - model = {test_fitness[0] - bh_fitness}')
        ind1 = str(ind).replace(',', '-')
        tmp = f'{trend},{stock_name},{train_fitness},{test_fitness[0]},{bh_fitness},20,{ind1},{consts}\n'
        with open(f'./output/outputup-PSO_dim5-multipier-selTour-{const_multiplier}-itter-{max_pso_itter}.csv', 'a') as fd:
            fd.write(tmp)

    report()

if __name__ == "__main__":
    stock_name = str(sys.argv[1])
    stock_trend = str(sys.argv[2])
    const_multiplier=int(sys.argv[3])
    max_pso_itter=int(sys.argv[4])
    util.read_candles()
    run_gp(stock_name,stock_trend,const_multiplier,max_pso_itter)