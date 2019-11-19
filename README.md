# bayesdifflite
Bayesian inference of linear temporal logic (LTL) specifications to explain differences across two sets of traces.

## Main
This is the main script that takes in input from `input.json` and outputs top-10 contrastive explanations dumped to `output.json` (also printed to the console). 

### input description
`input.json` is a dictionary containing the following items:

* `vocab`: a list of **all relevant propositions** describing state of the world.
* `traces_pos`: a list of lists, where each list is an **execution trace** (using propositions from the vocabulary). Propositions assert what is true about the system at each time step. This is a "positive" cluster where induced explanations are supposed to be satisfied.
* `traces_neg`: Similar to `traces_pos`, but represents a "negative" cluster where induced explanations are expected to be **dissatisfied**.
* `params`: dictionary containing inference parameters
	* `conjoin`: [true/false], indicating whether or not the hypothesis space includes conjuncts.
	* `inference`: ['mh','brute'], representing Metropolis-Hasting sampling or delimited enumeration method, respectively. 
	* `iterations`: [integer], the number of inference iterations.
* `probs_templates`: prior probablities of LTL templates
    * This is optional. Uniform by default.

### output description
`output.json` is a list of dictionaries, where each dict describes an induced LTL contrastive explanation along with its contrastive validity score ("cscore"). 


### run
```
python3 main.py -d input.json
```

## Template Descriptions

| Formula | Props | Description  |
| ------------- |:-------------:| :---------| 
| `eventual`| p | p occurs (may later be false). |
| `global`| p | p is always true through the trace. |
| `stability` | p | p eventually occurs and remains true. |
| `atmostonce` | p | Only one contiguous interval exists where p is true. |
| `response` | p, q | If p occurs, q eventually follows. |
| `sometime-before` | p, q | If p occurs, q occurred in the past. |
| `until` | p, q | p has to remain true until q eventually occurs (then p is free).  |


## Notes
* Results (induced explanations) can be stochastic, because both Metropolis-Hastings and delimited enumeration methods are stochastic algorithms.


## Collaborators

* :email: [Joseph Kim (MIT, IBM)](mailto:joseph_kim@csail.mit.edu)
* :email: [Christian Muise (IBM)](mailto:christian.muise@gmail.com)
* :email: [Shubham Agarwal (IBM)](mailto:agarwalshubham2007@gmail.com)
* :email: [Mayank Agarwal (IBM)](mailto:mayank.agarwal@ibm.com)
