import random
import numpy as np
import pandas as pd
from deap import gp
from collections import defaultdict

def seed(n):
    random.seed(n)
    np.random.seed(n)

def read_candles():
    df = pd.read_csv('../Dataset/output/SP500.csv')
    df['Date'] = pd.to_datetime(df['Date'], yearfirst=True)
    global dfs
    dfs = {name: sdf.set_index('Date').sort_index() for name, sdf in df.groupby('Name')}

def read_candles_one(stock):
    return dfs[stock].drop(['Name'], axis=1)

# https://github.com/DEAP/deap/pull/191/files
__type__ = object

def cxOnePoint(ind1, ind2):
    """Randomly select in each individual and exchange each subtree with the
    point as root between each individual.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        # Not STGP optimization
        types1[__type__] = xrange(1, len(ind1))
        types2[__type__] = xrange(1, len(ind2))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        common_types = []
        for idx, node in enumerate(ind2[1:], 1):
            if node.ret in types1 and not node.ret in types2:
                common_types.append(node.ret)
            types2[node.ret].append(idx)

    if len(common_types) > 0:
        type_ = random.choice(common_types)

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2

def cxOnePointLeafBiased(ind1, ind2, termpb):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First typed tree participating in the crossover.
    :param ind2: Second typed tree participating in the crossover.
    :param termpb: The probability of chosing a terminal node (leaf).
    :returns: A tuple of two typed trees.

    When the nodes are strongly typed, the operator makes sure the
    second node type corresponds to the first node type.

    The parameter *termpb* sets the probability to choose between a terminal
    or non-terminal crossover point. For instance, as defined by Koza, non-
    terminal primitives are selected for 90% of the crossover points, and
    terminals for 10%, so *termpb* should be set to 0.1.
    """

    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # Determine wether we keep terminals or primitives for each individual
    terminal_op = partial(eq, 0)
    primitive_op = partial(lt, 0)
    arity_op1 = terminal_op if random.random() < termpb else primitive_op
    arity_op2 = terminal_op if random.random() < termpb else primitive_op

    # List all available primitive or terminal types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    common_types = []

    for idx, node in enumerate(ind1[1:], 1):
        if arity_op1(node.arity):
            types1[node.ret].append(idx)

    for idx, node in enumerate(ind2[1:], 1):
        if arity_op2(node.arity):
            if node.ret in types1 and not node.ret in types2:
                common_types.append(node.ret)
            types2[node.ret].append(idx)

    if len(common_types) > 0:
        type_ = random.choice(common_types)

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2

def DEAP_Monkey_Patching():
    gp.cxOnePoint = cxOnePoint
    gp.cxOnePointLeafBiased = cxOnePointLeafBiased