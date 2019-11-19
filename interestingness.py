import numpy as np
import ltlfunc


def get_reactive_constraint(name, props):
    """
    Returns a reactive constraint based on the LTL name and propositions
    :param name: LTL formula name (global, eventual, etc.) for which the reactive constraint is requested
    :param props: List of propositions involved in the LTL formula
    :return: a dictionary of RCon with keys: alpha, and psi.
            The resulting RCon is of the form ⍺ ↦ φ
            While ⍺ is a proposition, φ is a separated formula (φ_past, φ_present, φ_future)
    """

    """
    Based on the LTL template, the following RCon is returned
    
    φ_global:           t_start ↦ (True ⋀ True ⋀ ☐ p_i)
    φ_eventual:         t_start ↦ (True ⋀ True ⋀ ◇ p_i)
    φ_stability:        p_i ↦ (True ⋀ True ⋀ ☐ p_i)
    φ_response:         p_i ↦ (True ⋀ True ⋀ ◇ p_j)
    φ_until:            t_start ↦ (True ⋀ True ⋀ p_i U p_j)
    φ_atmostonce:       p_i ↦ (True ⋀ True ⋀ p_i U ( G ! p_i) )
    φ_sometime_before:  p_j ↦ (⎑ p_i ⋀ True ⋀ True)         
            # Slight discrepancy in the props notation for sometime-before. Refer to ltlfunc.py line 742 on how the 
              propositions are structured and what the LTL formula translates to.  
    """

    rcon = {
        "alpha": None,
        "psi": None
    }

    # Get the RCon trigger ⍺ for the LTL
    if name == "global":
        rcon["alpha"] = "t_start"
    elif name == "eventual":
        rcon["alpha"] = "t_start"
    elif name == "stability":
        rcon["alpha"] = props[0]
    elif name == "response":
        rcon["alpha"] = props[0]
    elif name == "until":
        rcon["alpha"] = "t_start"
    elif name == "atmostonce":
        rcon["alpha"] = props[0]
    elif name == "sometime_before":
        rcon["alpha"] = props[0]

    # Get the RCon separated formula (φ_past, φ_present, φ_future) for the LTL
    rcon["psi"] = get_LTLpf_separated(name, props)

    return rcon


def get_LTLpf_separated(name, props):
    """
    Returns the φ component (in separated form) of the RCon ⍺ ↦ φ
    :param name: LTL formula name (global, eventual, etc.)
    :param props: List of propositions on which the LTL is defined
    :return: A tuple of separated φ ≣ (φ_past, φ_present, φ_future)
    """

    # Separated formula contains past, present, and future temporal formula
    separated_formula = [None, None, None]

    if name == "global":
        # φ ≣ (True ⋀ True ⋀ ☐ p_i)
        separated_formula[0] = True
        separated_formula[1] = True
        separated_formula[2] = ['G', props[0]]

    elif name == "eventual":
        # φ ≣ (True ⋀ True ⋀ ◇ p_i)
        separated_formula[0] = True
        separated_formula[1] = True
        separated_formula[2] = ['F', props[0]]

    elif name == "stability":
        # φ ≣ (True ⋀ True ⋀ ☐ p_i)
        separated_formula[0] = True
        separated_formula[1] = True
        separated_formula[2] = ['G', props[0]]

    elif name == "response":
        # φ ≣ (True ⋀ True ⋀ ◇ p_j)
        separated_formula[0] = True
        separated_formula[1] = True
        separated_formula[2] = ['F', props[1]]

    elif name == "until":
        # φ ≣ (True ⋀ True ⋀ p_i U p_j)
        separated_formula[0] = True
        separated_formula[1] = True
        separated_formula[2] = ['U', props[0], props[1]]

    elif name == "atmostonce":
        # φ ≣ (True ⋀ True ⋀ p_i U ( G ! p_i) )
        separated_formula[0] = True
        separated_formula[1] = True
        separated_formula[2] = ['U', props[0], ['G', ['!', props[0] ] ] ]

    elif name == "sometime_before":
        # φ ≣ (⎑ p_i ⋀ True ⋀ True)
            # Past operators are verified by their mirror operators in the referred paper. Thus, using ◇ instead of ⎑
        separated_formula[0] = ['F', props[1]]
        separated_formula[1] = True
        separated_formula[2] = True

    return separated_formula


def get_activating_points(trace, rcon_alpha):
    """
    Returns a list of time instances where the trace is activated by RCon ⍺
    :param trace: One trace (a list of states in time)
    :param rcon_alpha: A proposition indicating the ⍺ of the RCon
    :return: a list of time instances where the trace is activated
    """

    activation_times = []

    if rcon_alpha == "t_start":
        activation_times = [float("-inf")]      # -∞ is used as a special symbol for t_start
    elif rcon_alpha == "t_end":
        activation_times = [float("inf")]       # +∞ is used as a special symbol for t_end
    else:
        activation_times = [i for i in range(len(trace)) if rcon_alpha in trace[i]]

    return activation_times


def get_subtraces(trace, i):
    """
    Trisects one trace at time point i into a list of (t_past, t_present, t_future)
    :param trace: One trace (a list of states in time)
    :param i: A time instance at which the trace is to be trisected
    :return: a tuple of (t_past, t_present, t_future)
    """

    if i == float("-inf"):
        # RCon alpha = t_start
        t_past = []
        t_present = []
        t_future = trace
    elif i == float("inf"):
        # RCon alpha = t_end
        t_past = trace
        t_present = []
        t_future = []
    else:
        t_past = trace[:i+1]
        t_present = trace[i:i+1]
        t_future = trace[i:]

    return (t_past, t_present, t_future)


def checkLTL(f, t, trace, vocab):
    """
    Checks the satisfaction of LTL formula `f` on trace `trace` at time step `t`
    :param f: An LTL formula (must be in TREE format using nested tuples)
    :param t: time stamp where `f` is to be evaluated
    :param trace: Execution trace. It should be a dictionary with 2 keys: 'name', and 'trace'
    :param vocab: A list of vocabulary of propositions
    :return: True/False indicating the satisfaction of LTL on trace
    """

    if type(f) == bool and f is True:
        # If the LTL is trivially True then return so
        return True
    elif len(trace) == 0:
        # Else if the trace is empty, return False
        return False

    return ltlfunc.checkLTL(f, t, trace, vocab)


def compute_interestingness(trace, ltl_name, ltl_props, vocab):
    """
    Computes the interestingess score for a one LTL formula (ltl_name & ltl_props) on trace
    :param trace: execution trace in the form of a list of states in time
    :param ltl_name: the LTL formula name (global, eventual, etc.) for which interestingness metric is required
    :param ltl_props: a list of propositions for the LTL for which interestingness metric is required
    :param vocab: a list of vocabulary of propositions
    :return: a float value indicating the interestingness metric for LTL defined by (ltl_name & ltl_props) on trace
    """

    rcon = get_reactive_constraint(ltl_name, ltl_props)
    activation_points = get_activating_points(trace, rcon['alpha'])
    fulfilled_points = []
    psi_past, psi_present, psi_future = rcon['psi']

    # If there are no activating points for the RCon, return interestingess = 0
    if len(activation_points) == 0:
        return 0

    # Iterate over all activation points, and verify the φ_past, φ_present, and φ_future on
    # past_trace, present_trace, and future_trace respectively at that time point i
    for i in activation_points:
        t_past, t_present, t_future = get_subtraces(trace, i)
        
        t_past = list(reversed(t_past))   # Reverse the past trace for evaluation


        # Check if the past trace is fulfilled by the RCon φ_past
        trace_past = {
            'name': 'past_trace',
            'trace': t_past
        }

        past_fulfillment = checkLTL(psi_past, t=0, trace=trace_past, vocab=vocab)

        # Check if the present trace is fulfilled by the RCon φ_present
        trace_present = {
            'name': 'present_trace',
            'trace': t_present
        }

        present_fulfillment = checkLTL(psi_present, t=0, trace=trace_present, vocab=vocab)

        # Check if the future trace is fulfilled by the RCon φ_future
        trace_future = {
            'name': 'future_trace',
            'trace': t_future
        }

        future_fulfillment = checkLTL(psi_future, t=0, trace=trace_future, vocab=vocab)

        # If past, present, and future traces are all fulfilled, the RCon is interestingly fulfilled at that time point
        if past_fulfillment and present_fulfillment and future_fulfillment:
            fulfilled_points.append(i)

    # interestingness = (no. of times RCon is interestingly fulfilled) / (no. of times the RCon is activated)
    interestingness = len(fulfilled_points) / float(len(activation_points))

    return interestingness


def compute_interestingness_conjunct(trace, ltl_name, ltl_props, vocab):
    """
    Computes the interestingness for a conjunct LTL over trace
    :param trace: execution trace in the form of a list of states over time
    :param ltl_name: the ltl formula name (global, eventual, etc.) for which interestingness metric is required
    :param ltl_props: a list of elements where each element is a list of propositions
                      these individual proposition lists are connected by conjunction ⋀
    :param vocab: a list of vocabulary of propositions
    :return: a float value indicating the interestingness of conjunct LTL over trace
    """

    interestingness_individual = [compute_interestingness(trace, ltl_name, props, vocab) for props in ltl_props]
    interestingness_conjunct = np.product(interestingness_individual)
    return interestingness_conjunct


def compute_support(traceset, ltl_name, ltl_props, vocab):
    """
    Returns the support metric of LTL over a trace set
    While interestingness is defined over a trace, the support is defined over the traceset
    and is a mean of interestingness scores over traces
    :param traceset: a list of traces, where each trace is dictionary of {'name', 'trace'} keys
    :param ltl_name: the ltl formula name (global, eventual, etc.)
    :param ltl_props: a list of elements with each element being a list of propositions
    :param vocab: a list of vocabulary of propositions
    :return: a float value indicating the support of LTL over traceset
    """

    if len(traceset) == 0:
        return 0

    interestingness_scores = [
        compute_interestingness_conjunct(trace['trace'], ltl_name, ltl_props, vocab) for trace in traceset
    ]

    support = np.mean(interestingness_scores)

    return support
