# Main script for inducing LTL contrastive explanations from input set of traces
#
# ARGUMENTS:
#   -d [input.json] : json file containing the required input (see README)
#
# OUTPUT:
#   output.json     : json output containing top-10 induced results
#
#
# Written by Joseph Kim

import argparse
from itertools import permutations
import json
from operator import itemgetter
import time

# Local imports
import ltlfunc
import interestingness

#############################################################


def run_ltl_inference(data, output_filepath=None):

    # Vocabulary (lowercased, unique)
    vocab = [s.lower() for s in data['vocab']]
    vocab = list(set(vocab))

    # Traces - organize both pos and neg clusters
    cluster_A = []
    for i, trace in enumerate(data['traces_pos']):
        trace = [[v.lower() for v in s] for s in trace]  # lowercase
        temp = dict()
        temp['name'] = 'a' + str(i)  # Create a name id
        temp['trace'] = tuple(trace)  # Use tuple
        cluster_A.append(temp)

    cluster_B = []
    for i, trace in enumerate(data['traces_neg']):
        trace = [[v.lower() for v in s] for s in trace]
        temp = dict()
        temp['name'] = 'b' + str(i)
        temp['trace'] = tuple(trace)
        cluster_B.append(temp)
    # X = (cluster_A, cluster_B)  # Evidence

    # Parameters
    inference = data['params']['inference']
    iterations = data['params']['iterations']
    conjoin = data['params']['conjoin']
    ltl_sample_cnt = data['params'].get('ltl_sample_cnt', 10)
    run_reversed_inference = data['params'].get('reversed_inference', True)
    verbose = data['params'].get('verbose', False)

    # Default inference parameters
    params = dict()
    params['alpha'] = data['params'].get('alpha', 0.01)
    params['beta'] = data['params'].get('beta', 0.01)
    params['lambda'] = data['params'].get('lambda', 0.60)
    params['epsilon'] = data['params'].get('epsilon', 0.2)

    # Print statistics
    print('\nSize of vocabulary = %s' % len(vocab))

    # Get LTL templates
    if 'probs_templates' in data:
        probs_templates = data['probs_templates']
    else:
        probs_templates = None
    templates = ltlfunc.getLTLtemplates(user_probs=probs_templates)

    # Get permutation tables
    perm_table = dict()
    perm_table[1] = [list(i) for i in permutations(vocab, 1)]
    perm_table[2] = [list(i) for i in permutations(vocab, 2)]

    ltl_rundata = [
        {'X': (cluster_A, cluster_B), 'reversed': False}
    ]

    if run_reversed_inference:
        ltl_rundata.append(
            {'X': (cluster_B, cluster_A), 'reversed': True}
        )

    # Preparing json output
    output_inference = list()

    for data_X in ltl_rundata:
        X = data_X['X']
        reversed = data_X['reversed']

        cluster_A_inf, cluster_B_inf = X

        output = list()

        print('*' * 50)
        print('Running LTL inference with reversed mode = %s' % str(reversed))
        print('Number of positive traces = %s' % len(cluster_A_inf))
        print('Number of negative traces = %s' % len(cluster_B_inf))

        #######################################################
        # RUN INFERENCE
        #
        # 1) Metropolis-Hastings Sampling
        if inference == 'mh':

            # Initial guess
            ltl_initial = ltlfunc.samplePrior(templates, vocab, perm_table, params['lambda'], conjoin)
            print('\n-Running MH..')
            print('-Initial guess = ' + ltl_initial['str_friendly'])
            st = time.time()

            # Preparation
            burn_in_mh = 500
            num_iter_mh = iterations + burn_in_mh
            memory = dict()
            cache = dict()

            # Run MH Sampler
            sampler = ltlfunc.MH_sampler(ltl_initial, X, vocab, templates, params, perm_table, memory, cache, conjoin)
            sampler.runMH(num_iter_mh, burn_in_mh, verbose=verbose)
            memory = sampler.memory

            # Print stats
            print('-MH runtime = ' + format(time.time() - st, '.3f'))
            print(
                '-MH number of accepted samples = %s / %s' % (sampler.num_accepts, len(sampler.accept_reject_history)))
            print('-MH number of perfect valid samples = %s / %s' %
                  (int(sum([j for j in sampler.cscore_history if j == 1])), num_iter_mh - burn_in_mh))

            # Rank posterior and print top-10 samples
            print('\n-MH Top-{} Specs'.format(ltl_sample_cnt))
            ranked = sorted(sampler.posterior_dict, key=sampler.posterior_dict.get, reverse=True)
            i = 0

            for r in ranked:
                cscore = sampler.cscore_dict[r]
                cscore1, cscore2 = memory[r]
                cscore2 = 1 - cscore2
                ltl_meaning = sampler.ltl_str_meanings[r]['meaning']

                ltl = sampler.ltl_log[r]
                ltl_name = ltl['name']
                ltl_props = ltl['props_list'] if conjoin else [ltl['props']]

                # Positive set support
                positive_support = interestingness.compute_support(cluster_A_inf, ltl_name, ltl_props, vocab)

                if positive_support == 0:
                    continue

                i += 1

                print('-' * 30)
                print(r, end='')
                print(' accuracy = %s' % cscore)
                print(' (individual scores): cscore1: %f, cscore2: %f' % (cscore1, cscore2))
                print(' Interestingness (support) : %f' % positive_support)
                print(' Meaning: %s' % ltl_meaning)

                if i >= ltl_sample_cnt:
                    break

                # Adding to output
                temp = dict()
                temp['formula'] = r
                temp['meaning'] = sampler.ltl_str_meanings[r]
                temp['accuracy'] = cscore
                temp['cscores_individual'] = (cscore1, cscore2)
                temp['interestingness'] = positive_support
                temp['reversed'] = reversed
                output.append(temp)



        # 2) Brute force search (delimited enumeration)
        elif inference == 'brute':

            print('\n-Running Brute Force Search')
            st = time.time()
            if conjoin:
                # Brute force random sampler (b/c pre-enumerating everything is intractable)
                print('-Collecting delimited set')
                ltl_full = []
                history = []
                num_brute_force = iterations

                # Collection loop
                while len(history) < num_brute_force:
                    s = ltlfunc.samplePrior(templates, vocab, perm_table, conjoin=conjoin, doRandom=True)
                    ltl_str = s['str_friendly']
                    if ltl_str not in history:
                        ltl_full.append(s)
                        history.append(ltl_str)
                print('-All delimited set collected. Time spent = ' + format(time.time() - st, '.3f'))


            else:
                # If not using conjunction, then obtain a full brute force list
                ltl_full = []
                for template in templates:
                    results = ltlfunc.instantiateLTLvariablePermutate(template, vocab)
                    ltl_full += results
                print('-Number of total possible LTL specs (no conjunctions): %s' % len(ltl_full))

            # Exact inference on collection
            memory = dict()
            cache = dict()
            for ltl_instance in ltl_full:
                log_posterior, cscore, memory = ltlfunc.computePosterior(ltl_instance, X, vocab, params, memory, cache,
                                                                         conjoin)
                ltl_instance['posterior'] = log_posterior
                ltl_instance['cscore'] = cscore
                ltl_instance['cscores_individual'] = memory[ltl_instance['str_friendly']]

            print('-Brute force collection and posterior collection complete. Time spent = ' + format(time.time() - st,
                                                                                                      '.3f'))

            # Rank posterior and print top-10 samples
            print('\n-Enumeration Top-{} Specs'.format(ltl_sample_cnt))
            ranked = sorted(ltl_full, key=itemgetter('posterior'), reverse=True)
            i = 0

            for r in ranked:
                cscore1, cscore2 = r['cscores_individual']
                cscore2 = 1 - cscore2

                ltl_name, ltl_props = r['name'], r['props_list']

                # Positive set support
                positive_support = interestingness.compute_support(cluster_A_inf, ltl_name, ltl_props, vocab)

                if positive_support == 0:
                    continue

                i += 1

                print('-' * 30)
                print(r['str_friendly'], end='')
                print(' accuracy = %s' % r['cscore'])
                print(' (individual scores): cscore1: %f, cscore2: %f' % (cscore1, cscore2))
                print(' Interestingness (support) : %f' % positive_support)
                print(' Meaning: %s' % r['str_meaning'])

                if i >= ltl_sample_cnt:
                    break

                # Adding to output
                temp = dict()
                temp['formula'] = r['str_friendly']
                temp['meaning'] = r['str_meaning']
                temp['accuracy'] = r['cscore']
                temp['cscores_individual'] = (cscore1, cscore2)
                temp['interestingness'] = positive_support
                temp['reversed'] = reversed
                output.append(temp)

        else:
            raise AttributeError("Wrong inference mode specified.")

        #######################################################
        # END OF INFERENCE
        #######################################################

        # Append local ltl order inference output to global output list
        output_inference.extend(output)
        output_inference = sorted(output_inference, key=lambda x: x['accuracy'], reverse=True)[:ltl_sample_cnt]

    # Dump output
    if output_filepath is not None:
        with open(output_filepath, 'w') as f:
            json.dump(output_inference, f, indent=4)

    return output_inference


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main script for LTL contrastive explanation induction')
    parser.add_argument('-d', type=str, required=True, help='input JSON file')
    parser.add_argument('-o', type=str, required=False, default='output.json', help='output JSON file')
    args = parser.parse_args()

    # Load input
    with open(args.d) as f:
        data = json.load(f)

    run_ltl_inference(data, args.o)






