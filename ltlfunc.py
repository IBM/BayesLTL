# Module on functions related to Linear Temporal Logic (LTL) - LITE version
# Written by Joseph Kim

import copy
import math
import numpy as np
import os
import random
import re
from itertools import permutations
from scipy.stats import poisson, geom
import sys


##############################
### BEGIN simple functions ###
def nCr(n,r):
    """ Returns number of possibilities for n choose r """
    f = math.factorial
    return f(n) // f(r) // f(n-r)


def rec_replace(l, d): 
    """ Recursively replace list l with values of dictionary d

        ARGUMENTS:
            l  -  list or list of lists
            d  -  dictionary where if an element in l matches d['key'], it is replaced by the value

        OUTPUT:
            l  -  list with elements updated
    """
    for i in range(len(l)):
        if isinstance(l[i], list):
            rec_replace(l[i], d)
        else:
            l[i] = d.get(l[i], l[i])
    return l



def to_tuple(lst):
    """ Convert an arbitrary nested list to nested tuple (recursively) """
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)



### END of simple functions ###
###############################


def computePrior(ltl, lamb = 0.9, conjoin = False):
    """ Returns the log-prior of current LTL sample

        INPUTS:
            ltl   - current ltl dict
            lamb    - lambda parameter for geometric distribution
            conjoin   - whether or not conjunctions of templates are being considered

        OUTPUT:
            log P(ltl)

     """

    # LTL template prior
    log_template = math.log(ltl['prob'])

    if conjoin:
        # Complexity based on number of conjunctions
        num_conjuncts = len(ltl['props_list'])
        complexity = geom.pmf(num_conjuncts, 1-lamb)
        # complexity = poisson(3).pmf(num_conjuncts)
        try:
            log_complexity = math.log(complexity)
        except ValueError:
            log_complexity = -1000
        return log_template + log_complexity
    else:
        return log_template


def samplePrior(templates, vocab, perm_table, lamb = 0.9, conjoin = False, doRandom = False):
    """ Samples a new ltl proportional to the prior distribution

        INPUTS:
            templates  - a list of LTL templates
            vocab      - vocabulary of propositions
            perm_table   - permutation table of vocab
            lamb         - lambda parameter for geometric distribution
            conjoin      - whether or not conjunctions of templates are considered
            doRandom     - whether or not to sample from pure uniform distribution

        OUTPUT:
            ltl      - a new sampled LTL dict

     """

    # Pick a template
    t = pickLTLtemplate(templates, usePrior = True)
    num_vars = len(t['vars'])

    if conjoin:

        # Pick number of conjunctions
        if not doRandom:

            # Geometric distribution
            num_conjuncts = geom.rvs(1-lamb)

            # Poisson distribution
            # num_conjuncts = poisson.rvs(3)

            if num_conjuncts == 0:
                num_conjuncts = 1
            elif num_conjuncts > len(perm_table[num_vars]):
                num_conjuncts = len(perm_table[num_vars])
        else:
            # Uniform over all possibilities
            if num_vars > 1:
                # Cut off at 100 max conjunctions
                num_conjuncts = random.randint(1, 100)
            else:
                num_conjuncts = random.randint(1, len(perm_table[num_vars]))


        # Put cap if it exceeds allotted number of conjunctions
        if num_conjuncts > len(perm_table[num_vars]):
            num_conjuncts = len(perm_table[num_vars])

        # Pick propositions
        props_list = random.sample(perm_table[num_vars], num_conjuncts)
        ltl = getLTLconjunct(t, vocab, props_list)

        return ltl

    else:
        ltl = instantiateLTL(t, vocab)
        return ltl



def computeLikelihood(ltl, X, vocab, params, memory = None, cache = None):
    """ Returns the log-likelihood of current ltl with respect to evidence X

        INPUTS:
            ltl  -  current ltl dict
            X    -  evidence (cluster1, cluster2) where each trace in cluster is a trace dict
            vocab    - vocabulary of propositions
            params   - params dict
            memory   - memory for checking ltl['str_friendly'] on X
            cache    - cache for checking LTL on subformulas
            conjoin  - whether or not this is for conjunction hypothesis space

        OUTPUT:
            log_likelihood  - log P(X | ltl)
            cscore          - [0-1], contrastive explanation validity score
    """

    beta = params['beta']
    alpha = params['alpha']
    cluster1 = X[0]
    cluster2 = X[1]

    # Check memory or calculate
    if type(memory) is dict and ltl['str_friendly'] in memory:
        cscore1, cscore2 = memory[ltl['str_friendly']]
    else:
        cscore1, cscore2  = checkContrastiveValidity(ltl, cluster1, cluster2, vocab, cache)
        if type(memory) is dict:
            memory[ltl['str_friendly']] = (cscore1, cscore2)
    # cscore = cscore1 * (1-cscore2)

    # Likelihood in terms of product over satisfaction/dissatisfaction of traces
    num_satisfy_1 = int(cscore1 * len(cluster1))
    num_fail_1 = len(cluster1) - num_satisfy_1
    num_satisfy_2 = int(cscore2 * len(cluster2))
    num_fail_2 = len(cluster2) - num_satisfy_2

    # cscore = (total positive entailment + total negative non-entailment) / (total positive and negative traces)
    cscore = (num_satisfy_1 + num_fail_2) / float(len(cluster1) + len(cluster2))

    log_likelihood = num_satisfy_1 * (1-alpha) + num_fail_1 * alpha + num_satisfy_2 * beta + num_fail_2 * (1-beta)

    # ALTERNATIVE LIKELIHOOD - Using flat degree of satisfaction
    # # Case 1) c1 SATISFY,  c2 NOT
    # c1 = cscore1 * (1-cscore2) * (1 - 2 * beta - gamma)
    # # Case 2) c1 NOT, c2 NOT
    # c2 = (1-cscore1) * (1-cscore2) * beta
    # # Case 3) c1 SATISFY, c2 SATISFY
    # c3 = cscore1 * cscore2 * beta
    # # Case 4) c1 NOT, c2 SATISFY
    # c4 = (1-cscore1) * cscore2 * gamma
    # likelihood = c1 + c2 + c3 + c4
    # log_likelihood = math.log(likelihood)

    return log_likelihood, cscore, memory


def computePosterior(ltl, X, vocab, params, memory = None, cache = None, conjoin = False):
    """ Returns the log-posterior and the current ltl's validity score on X 
        
        INPUTS:
            ltl  - ltl dict
            X    - evidence (cluster1, cluster2) where each trace in cluster is a trace dict
            vocab    - vocab of propositions
            params   - params dict
            memory   - memory for checking LTL based on ltl['str_friendly']
            cache    - cache for checking LTL on subformulas
            conjoin  - whether or not this is for conjunction space

        OUTPUT:
            log_posterior - log P(ltl | X)
            cscore    - contrastive validity score [0-1]

    """
    log_prior = computePrior(ltl, params['lambda'], conjoin)
    log_likelihood, cscore, memory = computeLikelihood(ltl, X, vocab, params, memory, cache)
    log_posterior = log_prior + log_likelihood
    return log_posterior, cscore, memory


def checkLTL(f, t, trace, vocab, c = None, v = False):
    """ Checks satisfaction of a LTL formula on an execution trace

        NOTES:
        * This works by using the semantics of LTL and forward progression through recursion
        * Note that this does NOT require using any off-the-shelf planner

        ARGUMENTS:
            f       - an LTL formula (must be in TREE format using nested tuples
                      if you are using LTL dict, then use ltl['str_tree'])
            t       - time stamp where formula f is evaluated
            trace   - execution trace (a dict containing:
                        trace['name']:    trace name (have to be unique if calling from a set of traces)
                        trace['trace']:   execution trace (in propositions format)
                        trace['plan']:    plan that generated the trace (unneeded)
            vocab   - vocabulary of propositions
            c       - cache for checking LTL on subtrees
            v       - verbosity

        OUTPUT:
            satisfaction  - true/false indicating ltl satisfaction on the given trace
    """

    if v: 
        print('\nCurrent t = '+str(t))
        print('Current f =',f)

    ###################################################

    # Check if first operator is a proposition
    if type(f) is str and f in vocab:
        return f in trace['trace'][t]

    # Check if sub-tree info is available in the cache
    key = (f, t, trace['name'])
    if c is not None:
        if key in c:
            if v: print('Found subtree history')
            return c[key]

    # Check for standard logic operators
    if f[0] in ['not', '!']: 
        value = not checkLTL(f[1], t, trace, vocab, c, v)
    elif f[0] in ['and', '&', '&&']:
        value = all( ( checkLTL(f[i], t, trace, vocab, c, v) for i in range(1,len(f)) ) )
    elif f[0] in ['or', '||']:
        value = any( ( checkLTL(f[i], t, trace, vocab, c, v) for i in range(1,len(f)) ) )
    elif f[0] in ['imp', '->']:
        value = not(checkLTL(f[1], t, trace, vocab, c, v)) or checkLTL(f[2], t, trace, vocab, c, v)

    # Check if t is at final time step
    elif t == len(trace['trace'])-1:
        # Confirm what your interpretation for this should be.
        if f[0] in ['X', 'G', 'F']: 
            value = checkLTL(f[1], t, trace, vocab, c, v)  # Confirm what your interpretation here should be
        elif f[0] == 'U': 
            value = checkLTL(f[2], t, trace, vocab, c, v) 
        elif f[0] == 'W':   # weak-until
            value = checkLTL(f[2], t, trace, vocab, c, v) or checkLTL(f[1], t, trace, vocab, c, v)
        elif f[0] == 'R':   # release (weak by default)
            value = checkLTL(f[2], t, trace, vocab, c, v)
        else:
            # Does not exist in vocab, nor any of operators
            sys.exit('LTL check - something wrong')

    else:
    # Forward progression rules
        if f[0] == 'X': 
            value = checkLTL(f[1], t+1, trace, vocab, c, v)
        elif f[0] == 'G': 
            value = checkLTL(f[1], t, trace, vocab, c, v) and checkLTL(('G',f[1]), t+1, trace, vocab, c, v)
        elif f[0] == 'F': 
            value = checkLTL(f[1], t, trace, vocab, c, v) or checkLTL(('F',f[1]), t+1, trace, vocab, c, v)
        elif f[0] == 'U':
            # Basically enforces f[1] has to occur for f[1] U f[2] to be valid.
            if t == 0:
                if not checkLTL(f[1], t, trace, vocab, c, v):
                    value = False
                else:
                    value = checkLTL(f[2], t, trace, vocab, c, v) or (checkLTL(f[1], t, trace, vocab, c, v) and checkLTL(('U',f[1],f[2]), t+1, trace, vocab, c, v))
            else:
                value = checkLTL(f[2], t, trace, vocab, c, v) or (checkLTL(f[1], t, trace, vocab, c, v) and checkLTL(('U',f[1],f[2]), t+1, trace, vocab, c, v))

        elif f[0] == 'W':  # weak-until
            value = checkLTL(f[2], t, trace, vocab, c, v) or (checkLTL(f[1], t, trace, vocab, c, v) and checkLTL(('W',f[1],f[2]), t+1, trace, vocab, c, v))
        elif f[0] == 'R':  # release (weak by default)
            value = checkLTL(f[2], t, trace, vocab, c, v) and (checkLTL(f[1], t, trace, vocab, c, v) or checkLTL(('R',f[1],f[2]), t+1, trace, vocab, c, v))
        else:
            # Does not exist in vocab, nor any of operators
            sys.exit('LTL check - something wrong')
    
    if v: print('Returned: '+str(value))

    # Save result
    if c is not None and type(c) is dict:
        key = (f, t, trace['name'])
        c[key] = value  # append

    return value




def checkContrastiveValidity(ltl, cluster1, cluster2, vocab, cache = None):
    """ Computes constrastive explanation score of current ltl on given pair of traces 

        Where cluster1 is expected to satisfy ltl and cluster2 to dissatisfy it

        Note that with clusters, there exists a degree of satisfaction by the number of 
        satisfactory/unsatisfactory cases.

        ARGUMENTS:
            ltl  -  ltl dict
            cluster1, cluster2  - cluster [list] of traces, where each trace is a dict containing
                trace['name']:    trace name
                trace['trace']:   execution trace (in propositions format)
                trace['plan']:    generating plan of a trace (unneeded)
            vocab   - vocabulary of propositions
            cache   - cache for checking LTL on subtrees

        OUTPUT:
            cscore1, cscore2     - validity scores [0-1] for each cluster
    """

    ltl_tuple = to_tuple(ltl['str_tree'])

    cscore1 = 0
    for trace in cluster1:
        cscore1 += checkLTL(ltl_tuple, 0, trace, vocab, cache)
    cscore1 = 1. * cscore1 / len(cluster1)

    cscore2 = 0
    for trace in cluster2:
        cscore2 += checkLTL(ltl_tuple, 0, trace, vocab, cache)
    cscore2 = 1. * cscore2 / len(cluster2)

    return cscore1, cscore2



def instantiateLTLvariablePermutate(template, vocab):
    """ Instantiate a LTL template with a permutation of propositions. 
        Returns a list containg all possible permutations

        ARGUMENTS:
            template    - a ltl template
            vocab       - a set of propositions to draw permutation samples from

        OUTPUT:
            ltl_list    - a list of instantiated LTL dicts over all permutations of vocab
    """

    ltl_list = []
    num_vars = len(template['vars'])

    # Permutation of propositions of length equal to num_vars
    props_perm = permutations(vocab, num_vars)
    for p_tuple in props_perm:
        template_copy = template.copy()
        template_copy['props'] = list(p_tuple)
        ltl = produceLTLstring(template_copy) 
        ltl_list.append(ltl)

    return ltl_list



def instantiateLTL(template, vocab, props = None):
    """ Instantiate a LTL template with a randomly sampled propositions

        ARGUMENTS:
            template        - a ltl template
            vocab           - a set of propositions to draw samples from if props is None
            props           - propositions to instantiate with

        OUTPUT:
            ltl - a dict with following key/values
                'fml':    uninstantiated formula
                'ids':    indices where propositions should be inserted
                'prob' :  prior probability for the LTL template
                'props' : instantiated propositions [list]
                'str':    full instantiated string
    """

    ltl = template.copy()
    num_vars = len(ltl['vars'])
    if props:
        assert len(props) == num_vars
        ltl['props'] = props
    else:
        ltl['props'] = random.sample(vocab, num_vars)
    ltl = produceLTLstring(ltl)  # Fills in ltl['str']

    return ltl


def produceLTLStringMeaning(name, props_list):
    """
    Produces a natural language explanation of the LTL
    :param name: a string specifying the LTL template name
    :param props_list: a list of list of strings indicating the propositions
    :return: a string of natural language meaning of the LTL
    """

    connector_str = ' AND '
    template_map = {
        "global": '("{}") is true throughout the entire trace',
        "eventual": '("{}") eventually occurs (may later become false)',
        "stability": '("{}") eventually occurs and stays true forever',
        "response": 'If ("{}") occurs, ("{}") eventually follows',
        "until": '("{}") has to be true until ("{}") eventually becomes true',
        "atmostonce": 'Only one contiguous interval exists where ("{}") is true',
        "sometime_before": 'If ("{}") occurs, ("{}") occured in the past'
    }

    str_template = template_map[name]
    ltl_meaning = connector_str.join([str_template.format(*x) for x in props_list])

    return ltl_meaning


def produceLTLstring(ltl):
    """ Using the current ltl formula, ltl['fml'] and 
        the current propositions, ltl['props'], fills in the followings:
            ltl['str']:    formula string
            ltl['str_friendly']: formula string in friendly read form
            ltl['str_tree']:    formula string in tree mode

        NOTES:
            * Number of props in ltl['props'] must equal to that allowed by current template

        ARGUMENTS:
            ltl           - current LTL dict

        OUTPUT:
            ltl           - updated LTL dict 
                            (ltl['str'] and ltl['str_tree'] and ltl['str_friendly'])
    """

    ltl['str'] = ltl['fml'][:]
    num_vars = len(ltl['ids'])
    num_props = len(ltl['props'])

    # Check number of variables
    if num_vars != num_props:
        print('num_vars = %s' % num_vars)
        print('num_props = %s' % num_props)
        sys.exit('LTL props does NOT equal to length of variables in template.')

    # Fill in the string - ltl['str']
    for i, p in enumerate(ltl['props']):
        for j in ltl['ids'][i]:
            ltl['str'][j] = '"'+p+'"'
    ltl['str'] = ' '.join(ltl['str'])

    # Friendly string (using name header and current props)
    ltl['str_friendly'] = ltl['name'] + ': ' + ' , '.join(ltl['props'])
    ltl['str_meaning'] = produceLTLStringMeaning(ltl['name'], [ltl['props']])

    # Fill in the string (tree mode) - ltl['str_tree']
    rep_dict = dict()
    for i in range(num_vars):
        rep_dict[ltl['vars'][i]] = ltl['props'][i]
    tempcopy = copy.deepcopy(ltl['fml_tree'])
    ltl['str_tree'] = rec_replace(tempcopy, rep_dict)

    return ltl


def getLTLtemplates(choice = None, user_probs = None):
    """ Returns a list of LTL templates (uninstantiated), where each template t 
        is a dictionary following key/values:

        t['name']: LTL pattern name (str)
        t['fml']: LTL formula (uninstantiated) --> useful for ltlfond2fond
        t['ids']: indices where instantiated proposition(s) should go in t['fml']
        t['probs']: prior probability on template t
        t['vars']: number of free variables for each primitive LTL template
        t['fml_tree']: LTL formula is a tree structure

        INPUT:
            choice      -   a list of patterns (str) to include in templates

        OUTPUT:
            templates   -   a list of LTL dicts

    """
    templates = list()

    # Default template priors
    probs = dict()
    probs['eventual'] = 1.0
    probs['eventual_neg'] = 1.0
    probs['global'] = 1.0
    probs['global_neg'] = 1.0
    probs['until'] = 1.0
    probs['until_neg'] = 1.0

    probs['response'] = 1.0
    probs['response_neg'] = 1.0
    probs['response_strong'] = 1.0
    probs['response_strong_neg'] = 1.0
    probs['stability'] = 1.0
    probs['stability_strong'] = 1.0

    probs['atmostonce_strong'] = 1.0
    probs['atmostonce'] = 1.0
    probs['sometime_before'] = 1.0
    probs['sometime_before_strong'] = 1.0

    # Override with user prior probs
    if user_probs:
        probs.update(user_probs)

    # Default choice of templates
    if choice is None:
        choice = ['eventual', 'global', 'until',
                    'response', 'stability', 'atmostonce',
                    'sometime_before']


    #### T1: Eventually: v1 becomes true at some point
    t = dict()
    t['name'] = 'eventual'
    t['fml'] = ['F', '"v1"']
    t['vars'] = [ 'v1' ]
    t['ids'] = [ [1] ]
    t['prob'] = probs['eventual']
    t['fml_tree'] = ['F', 'v1']
    if t['name'] in choice: templates.append(t)

    # # Negation
    # t = dict()
    # t['name'] = 'eventual_neg'
    # t['fml'] = ['! F', '"v1"']
    # t['vars'] = [ 'v1' ]
    # t['ids'] = [ [1] ]
    # t['prob'] = probs['eventual_neg']
    # t['fml_tree'] = ['not', ['F', 'v1']]
    # if t['name'] in choice: templates.append(t)
    # ####

    #### T2: Global: v1 is true always
    t = dict()
    t['name'] = 'global'
    t['fml'] = ['G', '"v1"']
    t['vars'] = [ 'v1' ]
    t['ids'] = [ [1] ]
    t['prob'] = probs['global']
    t['fml_tree'] = ['G', 'v1']
    if t['name'] in choice: templates.append(t)

    # # Negation
    # t = dict()
    # t['name'] = 'global_neg'
    # t['fml'] = ['! G', '"v1"']
    # t['vars'] = [ 'v1' ]
    # t['ids'] = [ [1] ]
    # t['prob'] = probs['global_neg']
    # t['fml_tree'] = ['not', ['G', 'v1']]
    # if t['name'] in choice: templates.append(t)
    # #### 

    #### T3: Until: 
    # v1 is true until v2 becomes true (v2 has to become true at some point)
    # after v2 becomes true, v1 is unrestricted
    t = dict()
    t['name'] = 'until'
    t['fml'] = ['"v1"', 'U', '"v2"']
    t['vars'] = [ 'v1', 'v2' ]
    t['ids'] = [ [0], [2] ]
    t['prob'] = probs['until']
    t['fml_tree'] = ['U', 'v1', 'v2']
    if t['name'] in choice: templates.append(t)

    # # Negation
    # t = dict()
    # t['name'] = 'until_neg'
    # t['fml'] = ['!(', '"v1"', 'U', '"v2"', ')']
    # t['vars'] = [ 'v1', 'v2' ]
    # t['ids'] = [ [1], [3] ]
    # t['prob'] = probs['until_neg']
    # t['fml_tree'] = ['not', ['U', 'v1', 'v2']]
    # if t['name'] in choice: templates.append(t)
    # ####

    #### T4: Response: 
    # Globally, if v1 occurs, eventually v2 occurs.
    # (If v1 does not ever occur, this is true)
    t = dict()
    t['name'] = 'response'
    t['fml'] = ['G (', '"v1"', '-> X F', '"v2"', ')']
    t['vars'] = [ 'v1', 'v2']
    t['ids'] = [ [1], [3] ]
    t['prob'] = probs['response']
    t['fml_tree'] = ['G', ['imp', 'v1', ['X', ['F', 'v2'] ] ] ]
    if t['name'] in choice: templates.append(t)

    # # Negation
    # t = dict()
    # t['name'] = 'response_neg'
    # t['fml'] = ['! ( G (', '"v1"', '-> X F', '"v2"', ') )']
    # t['vars'] = [ 'v1', 'v2']
    # t['ids'] = [ [1], [3] ]
    # t['prob'] = probs['response_neg']
    # t['fml_tree'] = ['not', ['G', ['imp', 'v1', ['X', ['F', 'v2'] ] ] ] ]
    # if t['name'] in choice: templates.append(t)
    # ####


    # #### T5: Response (strong): 
    # # Globally, if v1 occurs, eventually v2 occurs.
    # # (Enforces the occurrence of v1)
    # t = dict()
    # t['name'] = 'response_strong'
    # t['fml'] = ['F', '"v1"', '&& G (', '"v1"', '-> X F', '"v2"', ')']
    # t['vars'] = [ 'v1', 'v2']
    # t['ids'] = [ [1,3], [5] ]
    # t['prob'] = probs['response_strong']
    # t['fml_tree'] = ['and', ['F', 'v1']  ,  ['G', ['imp', 'v1', ['X', ['F', 'v2'] ] ] ] ]
    # if t['name'] in choice: templates.append(t)

    # # Negation
    # t = dict()
    # t['name'] = 'response_strong_neg'
    # t['fml'] = ['! ( F', '"v1"', '&& G (', '"v1"', '-> X F', '"v2"', ') )']
    # t['vars'] = [ 'v1', 'v2']
    # t['ids'] = [ [1,3], [5] ]
    # t['prob'] = probs['response_strong_neg']
    # t['fml_tree'] = ['not', ['and', ['F', 'v1']  ,  ['G', ['imp', 'v1', ['X', ['F', 'v2'] ] ] ] ] ]
    # if t['name'] in choice: templates.append(t)
    # ####

    #### T6: Eventually occurs and stays true forever
    t = dict()
    t['name'] = 'stability'
    t['fml'] = ['F G (', '"v1"',')']
    t['vars'] = ['v1']
    t['ids'] = [ [1] ]
    t['prob'] = probs['stability']
    t['fml_tree'] = ['F', ['G', 'v1'] ]
    if t['name'] in choice: templates.append(t)


    # #### T6: Once v1, always v1
    # t = dict()
    # t['name'] = 'stability'
    # t['fml'] = ['G (', '"v1"', '-> G', '"v1"',')']
    # t['vars'] = ['v1']
    # t['ids'] = [ [1,3] ]
    # t['prob'] = probs['stability']
    # t['fml_tree'] = ['G', ['imp', 'v1', ['G', 'v1'] ] ]
    # if t['name'] in choice: templates.append(t)

    # # Negation
    # t = dict()
    # t['fml'] = ['! ( G (', '"v1"', '-> G', '"v1"',') )']
    # t['vars'] = ['v1']
    # t['ids'] = [ [1,3] ]
    # t['prob'] = probs['stability_neg']
    # t['fml_tree'] = ['not', ['G', ['imp', 'v1', ['G', 'v1'] ] ] ]
    # if onmode['stability__neg']: templates.append(t)
    ####


    # #### T7: Once v1, always v1 (strong)
    # t = dict()
    # t['name'] = 'stability_strong'
    # t['fml'] = ['F', '"v1"', '&& G (', '"v1"', '-> G', '"v1"',')']
    # t['vars'] = ['v1']
    # t['ids'] = [ [1,3,5] ]
    # t['prob'] = probs['stability_strong']
    # t['fml_tree'] = ['and', ['F', 'v1'], ['G', ['imp', 'v1', ['G', 'v1'] ] ] ]
    # if t['name'] in choice: templates.append(t)
    # ####


    #### T8: At most once
    # -IF v1 becomes true and then stays true and then (possibly) becomes false and stays false
    # -There exists only one interval in the plan over which v1 is true.
    t = dict()
    t['name'] = 'atmostonce'
    t['fml'] = ['G (', '"v1"', '-> (', '"v1"', ' W (G ( !', '"v1"', '))))']
    t['vars'] = ['v1']
    t['ids'] = [ [1, 3, 5] ]
    t['prob'] = probs['atmostonce']
    t['fml_tree'] = ['G', ['imp', 'v1', 
                                  ['W', 'v1', 
                                        ['G', ['not', 'v1']]
                                   ]
                           ]
                    ]
    if t['name'] in choice: templates.append(t)


    # #### T8: At most once (STRONG)
    # # -v1 becomes true and then stays true and then (possibly) becomes false and stays false
    # # -There has to exist only one interval in the plan over which v1 is true.
    # t = dict()
    # t['name'] = 'atmostonce_strong'
    # t['fml'] = ['F', '"v1"', '&& G (', '"v1"', '-> (', '"v1"', ' U (G ( !', '"v1"', '))))']
    # t['vars'] = ['v1']
    # t['ids'] = [ [1, 3, 5, 7] ]
    # t['prob'] = probs['atmostonce_strong']
    # t['fml_tree'] = ['and',
    #                     ['F', 'v1'], 
    #                     ['G', ['imp', 'v1', 
    #                                   ['U', 'v1', 
    #                                         ['G', ['not', 'v1']]
    #                                    ]
    #                            ]
    #                     ]
    #                 ]
    # if t['name'] in choice: templates.append(t)

    #### T9: Sometime-before (v1, v2):
    # -If v1 occurs, sometime before v1 (no overlap), v2 must have occurred
    # -v2 has to occur before A occurs
    # -Doesn't enforce v2 or v1 to occur
    t = dict()
    t['name'] = 'sometime_before'
    t['fml'] = ['( ', '"v2"', ' && ! ', '"v1"', ') R ( ! ', '"v1"', ' )']
    t['vars'] = ['v1', 'v2']
    t['ids'] = [ [3, 5], [1] ]
    t['prob'] = probs['sometime_before']
    t['fml_tree'] = ['R', 
                        ['and', 'v2', ['not', 'v1']],
                        ['not', 'v1']
                    ]
    if t['name'] in choice: templates.append(t)


    # #### T9: Sometime-before (strong):
    # # Same as above but enforces v1 to occur
    # t = dict()
    # t['name'] = 'sometime_before_strong'
    # t['fml'] = ['F ', '"v1"', ' && (( ', '"v2"', ' && ! ', '"v1"', ') R ( ! ', '"v1"', ' ))']
    # t['vars'] = ['v1', 'v2']
    # t['ids'] = [ [1, 5, 7], [3] ]
    # t['prob'] = probs['sometime_before_strong']
    # t['fml_tree'] = ['and', ['F', 'v1'],
    #                         ['R', 
    #                             ['and', 'v2', ['not', 'v1']],
    #                             ['not', 'v1']
    #                         ]
    #                 ]
    # if t['name'] in choice: templates.append(t)


    # # Precedence: v1 always precedes v2
    # t = dict()
    # t['fml'] = ['"v1"', 'R (!', '"v2"', '||', '"v1"', ')']
    # t['ids'] = [ [0,4], [2] ]
    # t['prob'] = 1.5
    # templates.append(t)


    # Normalize the prior probabilities
    total_prob_mass = sum([t['prob'] for t in templates])
    for i in range(len(templates)):
        templates[i]['prob'] = 1. * templates[i]['prob'] / total_prob_mass

    return templates


def pickLTLtemplate(templates, current = None, change = False, name = None, usePrior = False):
    """ Randomly picks a LTL template from the list of templates

        ARGUMENTS:
            templates  - templates (list of LTL dicts)
            current    - current LTL dict
            change     - whether or not to pick new template that is different than the current template
            name       - directly specifying a template to pick
            usePrior   - pick a template proportional to their prior

        OUTPUT:
            t          - a new LTL template (uninstantiated)
    """

    if usePrior:
        probs = [t['prob'] for t in templates]
        return np.random.choice(templates, 1, p = probs)[0]

    if current is None:
        if name:
            possibles = [t for t in templates if t['name'] == name]
            return random.choice(possibles)
        else:
            return random.choice(templates)

    if change is True:
        possibles = [t for t in templates if t['name'] != current['name']]
        return random.choice(possibles)




def getLTLconjunct(t, vocab, props_list):
    """ Returns ltl which is a conjunction of ltl template t instantiated with list from
        props_list

        INPUTS:
            t            - ltl template over where conjunction is applied
            vocab        - full vocabulary
            props_list   - input list of proposition(s) to include in conjunctions

        OUTPUT:
            t_conjunct   -  new ltl dict (conjunct)

        NOTES:
            * Currently populates everything except ['vars'] and ['ids']

    """

    # Check that each value in props matches the number of free vars in template t
    assert len(props_list[0]) == len(t['vars'])

    # Sort the props_list alphabetically
    props_list.sort(key=lambda x: str(' '.join(x)))

    # Begin
    t_conjunct = dict()
    t_conjunct['name'] = t['name']
    t_conjunct['prob'] = t['prob']
    t_conjunct['str_tree'] = ['and']
    t_conjunct['str_friendly'] = t_conjunct['name'] + ': '
    t_conjunct['str'] = ''
    t_conjunct['str_meaning'] = ''
    p_set = set()

    # Loop through each conjunction
    for i, p in enumerate(props_list):
        ltl = instantiateLTL(t, vocab, p)
        p_set.update(set(p))
        t_conjunct['str_tree'] += [ltl['str_tree']]

        if i == 0:
            t_conjunct['str_friendly'] += '('+ ','.join(p) + ')'
            t_conjunct['str'] += '(' + ltl['str'] + ')'
        else:
            t_conjunct['str_friendly'] += ', ('+ ','.join(p) + ')'
            t_conjunct['str'] += ' && ' + '(' + ltl['str'] + ')'

    
    # Get ['props']
    t_conjunct['props'] = list(p_set)

    # Create ['props_list']
    t_conjunct['props_list'] = props_list

    # Construct ['fml']
    fml = t_conjunct['str']
    for i, p in enumerate(t_conjunct['props']):
        # Replace p in 'str'
        pattern = '"'+p+'"'
        replace = '"v' + str(i+1) +'"'
        fml = re.sub(pattern, replace, fml)
    t_conjunct['fml'] = fml

    # Create LTL meaning
    t_conjunct['str_meaning'] = produceLTLStringMeaning(t_conjunct['name'], t_conjunct['props_list'])


    return t_conjunct


######################################################################
######################################################################
######################################################################
    

class MH_sampler():
    """ Metropolis-Hastings Sampler """
    def __init__(self, ltl_initial, X, vocab, templates, params, perm_table, memory = None, cache = None, conjoin = False):
        self.ltl_old = ltl_initial
        self.X = X
        self.vocab = vocab
        self.templates = templates
        self.params = params
        self.perm_table = perm_table
        self.conjoin = conjoin
        self.memory = memory
        self.cache = cache
        self.posterior_dict = dict()
        self.cscore_dict = dict()

        # Recording
        self.ltl_samples = []
        self.ltl_log = {}
        self.accept_reject_history = []
        self.cscore_history = []
        self.best_cscore = 0
        self.best_cscore_history = []
        self.ltl_str_meanings = dict()

        # Probabilities
        self.log_posterior_old, self.ltl_old['cscore'], self.memory = computePosterior(ltl_initial, X, vocab, params, memory, self.cache, conjoin)
        self.posterior_dict[ltl_initial['str_friendly']] = self.log_posterior_old
        self.cscore_dict[ltl_initial['str_friendly']] = self.ltl_old['cscore']


    def addConjunct(self, t):

        num_vars = len(t['vars'])

        # Randomly pick a new conjunction to add
        possibles = [p for p in self.perm_table[num_vars] if p not in self.ltl_old['props_list']]
        p = random.choice(possibles)

        # Return new ltl      
        ltl = getLTLconjunct(t, self.vocab, self.ltl_old['props_list'] + [p] )

        return ltl


    def removeConjunct(self, t):

        # Remove one conjunction
        props_list = random.sample(self.ltl_old['props_list'], len(self.ltl_old['props_list']) - 1)

        # Return new ltl
        ltl = getLTLconjunct(t, self.vocab, props_list)

        return ltl


    def moveLTLconjoin(self, epsilon = 0.1):
        """ Proposal kernel for ltl conjunction space

            NOTES:
                * Features the following main moves:
                    -Sample from prior
                    -Drift from incumbent
                        -Add a conjunction
                        -Remove a conjunction
                * The two main moves (prior vs drift) is selected based on flat exploration schedule, epsilon
                    (for future work, consider making it adaptive, e.g. using current validity score)

            INPUTS:
                epsilon  - exploration schedule [0-1] denoting when to sample from prior
                            instead of using drifting from incumbent

            OUTPUTS:
                ltl                -  perturbed ltl
                transition_prob    - transition ratio, P(old|new) / P(new|old)
        """


        # Sample from the prior
        if random.random() < epsilon:

            ltl = samplePrior(self.templates, self.vocab, self.perm_table, self.params['lambda'], conjoin = True)
            transition_forward = 1
            transition_backward = 1

        # Drift kernel
        else:

            # Current template
            t = pickLTLtemplate(self.templates, name = self.ltl_old['name'])
            num_vars = len(t['vars'])
            num_conjuncts_incumbent = len(self.ltl_old['props_list'])
            num_all_perms = len(self.perm_table[num_vars])


            # Case when you can't add anymore
            if num_conjuncts_incumbent == num_all_perms:

                # Remove
                ltl = self.removeConjunct(t)
                transition_forward = 1. / num_conjuncts_incumbent
                transition_backward = 1

            # Case when you can't remove anymore
            elif num_conjuncts_incumbent == 1:

                # Add
                ltl = self.addConjunct(t)
                transition_forward = 1. / (num_all_perms - num_conjuncts_incumbent)
                transition_backward = 1. / 2

            # All other regular cases
            else:

                # Add
                if random.random() < 0.5:
                    ltl = self.addConjunct(t)
                    transition_forward = 1. / (num_all_perms - num_conjuncts_incumbent)
                    transition_backward = 1. / (num_conjuncts_incumbent + 1)

                # Remove
                else:
                    ltl = self.removeConjunct(t)
                    transition_forward = 1. / num_conjuncts_incumbent
                    transition_backward = 1. / (num_all_perms - (num_conjuncts_incumbent - 1))


        # Transition
        transition_prob = transition_backward / transition_forward
        transition_prob = 1


        return ltl, transition_prob



    def moveLTL(self, prob = 0.5):
        """ Proposal kernel for flat ltl selection: sample from prior """

        ## 1) Sampling from prior
        ltl = samplePrior(self.templates, self.vocab, self.perm_table, self.params['lambda'])
        transition_prob = 1
        return ltl, transition_prob

        # ## 2) Inter-intra moves
        # # Intra- move
        # if random.random() < prob:
        #     num_props_old = len(self.ltl_old['props'])

        #     # Option 1) Randomly pick propositions
        #     props = random.sample(self.vocab, num_props_old)

        #     # # Option 2) Pick a new list of propositions (forcing new set)
        #     # arePropsSame = True
        #     # while arePropsSame:
        #     #     props = random.sample(self.vocab, num_props_old)
        #     #     if props != self.ltl_old['props']:
        #     #         arePropsSame = False

        #     # Transition
        #     transition_prob = 1

        #     # Update
        #     ltl = self.ltl_old.copy()
        #     ltl['props'] = props
        #     ltl = produceLTLstring(ltl)
        #     return ltl, transition_prob

        # # Inter- move
        # else:
        #     # Switch to a new template
        #     template = pickLTLtemplate(self.templates, current = self.ltl_old, change = True)

        #     # Option 1) Randomly instantiate
        #     ltl = instantiateLTL(template, self.vocab)
        #     transition_prob = 1

        #     # # Option 2) Try to keep the same propositions
        #     # ltl = template
        #     # ltl['props'] = self.ltl_old['props']
        #     # ltl = produceLTLstring(ltl, self.vocab)
        #     #
        #     # # Transition probability
        #     # n_old = len(self.ltl_old['props'])
        #     # n_new = len(ltl['props'])
        #     # if n_new > n_old:
        #     #     transition_forward = 1. / len(self.vocab) ** (n_new - n_old)      # P(s' | s)
        #     #     transition_backward = 1. / nCr(n_new, n_old)                      # P(s | s')
        #     #     transition_prob = transition_backward / transition_forward
        #     # elif n_old > n_new:
        #     #     transition_forward = 1. / nCr(n_old, n_new)     
        #     #     transition_backward = 1. / len(self.vocab) ** (n_old - n_new)  
        #     #     transition_prob = transition_backward / transition_forward
        #     # else:
        #     #     transition_prob = 1

        #     return ltl, transition_prob


    def runMH(self, num_iter, burn_in = 0, verbose = False):

        for i in range(num_iter):
            if verbose:
                print('MH iteration {}/{}'.format(i, num_iter-1))
                print('-Current LTL: '+ self.ltl_old['str_friendly'])
                # print(self.ltl_old['str_tree'])

            # Transition from current old
            if self.conjoin:
                ltl_new, transition_prob = self.moveLTLconjoin(self.params['epsilon'])
            else:
                ltl_new, transition_prob = self.moveLTL()
            if verbose: print('-New LTL: '+ ltl_new['str_friendly'])
                # print(ltl_new['str_tree'])


            # Compute acceptance ratio with posteriors and transition prob
            log_posterior_new, ltl_new['cscore'], self.memory = computePosterior(ltl_new, self.X, self.vocab, self.params, self.memory, self.cache, self.conjoin)
            self.posterior_dict[ltl_new['str_friendly']] = log_posterior_new
            self.cscore_dict[ltl_new['str_friendly']] = ltl_new['cscore']
            self.ltl_str_meanings[ltl_new['str_friendly']] = {
                'str': ltl_new['str'],
                'meaning': ltl_new['str_meaning']
            }
            self.ltl_log[ltl_new['str_friendly']] = ltl_new.copy()
            
            log_acceptance = log_posterior_new - self.log_posterior_old + math.log(transition_prob)
            acceptance_ratio = math.exp(min(0, log_acceptance))

            # Bookkeeping
            if i >= burn_in: 
                self.cscore_history.append(ltl_new['cscore'])
            if verbose: 
                print('--Contrastive validity score (old): {}'.format(self.ltl_old['cscore']))
                print('--Contrastive validity score (new): {}'.format(ltl_new['cscore']))
                # print('--Transition prob = {:0.2f}'.format(transition_prob))
                print('--Acceptance prob = {:0.2f}'.format(acceptance_ratio))


            # Accept or reject the new candidate
            prob = min(1, acceptance_ratio)
            if random.random() < prob:
                # Acceptance
                self.log_posterior_old = log_posterior_new
                self.ltl_old = ltl_new.copy()

                # Record best cscore
                if ltl_new['cscore'] > self.best_cscore:
                    self.best_cscore = ltl_new['cscore']

                # Record after burn_in
                if i >= burn_in:    
                    self.ltl_samples.append(ltl_new)
                    self.accept_reject_history.append(True)
                    if verbose: print('--ACCEPTED')
            else:
                # Rejected
                # -Do nothing

                # Record after burn_in
                if i >= burn_in:
                    self.accept_reject_history.append(False)
                    if verbose: print('--Rejected')

            self.best_cscore_history.append(self.best_cscore)
            if verbose: print('\n')

        # AFTER MH sampling
        self.num_accepts = sum(self.accept_reject_history)


        if verbose:
            print('MH number of accepted samples = %s / %s' % (self.num_accepts, len(self.accept_reject_history)))
            print('MH number of valid samples = %s / %s' % 
                (int(sum([i for i in self.cscore_history if i == 1])), num_iter - burn_in))
            # print('Accepted samples')
            # print(self.ltl_samples)


##############################################
##############################################
