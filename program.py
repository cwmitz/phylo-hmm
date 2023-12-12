import numpy as np

from functions import single_decoding_routine

## Definition of the parameters

# Nine animals from paper:
# Combining Phylogenetic and Hidden Markov Models in Biosequence Analysis
# Adam Siepel and David Haussler
# Journal of Computational Biology 2004 11:2-3, 413-428
animalNames = [
    "dog",
    "cat",
    "pig",
    "cow",
    "rat",
    "mouse",
    "baboon",
    "human",
    "chimp",
]
n_species = 9

# From base paper: "assuming the HKY substitution model and k = 10 states,
# we fitted a phylo-HMM to this alignment, obtaining an estimate of lambda = 0.94"
n_states = 10
# State-transition matrix
lmbda = 0.94
a = lmbda + 1 / n_states * (1 - lmbda)
b = 1 / n_states * (1 - lmbda)

# In formal definition of phylo-HMM A is a matrix of state-transition probabilities
# A = {a_j,k} where j, k = 1, ..., n_states and a_j,k = P(q_t = k | q_{t-1} = j)
A = b * np.ones((n_states, n_states))
for i in range(n_states):
    A[i, i] = a

# Alphabet of nucleotids
alphabet = ["A", "C", "T", "G"]
alphabetSize = 4
n_nucleotids = 1000000

# Initial-state probabilities: b_j = probabilility that state j is visited first
# (hypthesis: uniform distribution)
b = np.ones(n_states) / n_states

# Defining the phylogenetic model with topology and branch lengths from paper:
# Combining Phylogenetic and Hidden Markov Models in Biosequence Analysis
# Adam Siepel and David Haussler
# Journal of Computational Biology 2004 11:2-3, 413-428
tree = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: [],
    10: [{"branch": 0.1077, "node": 1}, {"branch": 0.0852, "node": 2}],
    11: [{"branch": 0.115, "node": 3}, {"branch": 0.1228, "node": 4}],
    12: [{"branch": 0.0847, "node": 5}, {"branch": 0.0767, "node": 6}],
    13: [{"branch": 0.0053, "node": 8}, {"branch": 0.0048, "node": 9}],
    14: [{"branch": 0.0331, "node": 7}, {"branch": 0.019, "node": 13}],
    15: [{"branch": 0.0567, "node": 10}, {"branch": 0.0392, "node": 11}],
    16: [{"branch": 0.322, "node": 12}, {"branch": 0.0939, "node": 14}],
    17: [{"branch": 0.0269, "node": 15}, {"branch": 0.0299, "node": 16}],
}

# Define scaling factors for each state-- r values in the paper)):
# Combining Phylogenetic and Hidden Markov Models in Biosequence Analysis
# Adam Siepel and David Haussler
# Journal of Computational Biology 2004 11:2-3, 413-428
scaling_factors = [0.6, 1.1, 1.8, 2.2, 2.5, 3.0, 3.2, 3.5, 3.9, 4.3]

# Background (eqilibrium) frequencies for each phylogenetic model-
# pi_(k,j) =  frequency of nucleotid j under state k
pi = np.array(
    [
        [
            0.23139251968997074,
            0.29229892176621347,
            0.23686638747675678,
            0.239442171067059,
        ],
        [
            0.2863517462686391,
            0.2541462983758936,
            0.20525472447594467,
            0.25424723087952267,
        ],
        [
            0.23763087044269857,
            0.312847933940467,
            0.24556233940029207,
            0.20395885621654244,
        ],
        [
            0.22038669448026416,
            0.24989123315243605,
            0.26200092899456406,
            0.26772114337273567,
        ],
        [
            0.30479468273151594,
            0.27419170797865045,
            0.2583575578469486,
            0.16265605144288517,
        ],
        [
            0.17326425988059183,
            0.3086404618172126,
            0.2586166120393451,
            0.2594786662628504,
        ],
        [
            0.32408836147815206,
            0.1829001750441552,
            0.2841236725590436,
            0.20888779091864917,
        ],
        [
            0.3002999878440192,
            0.21782391016484007,
            0.244709980031213,
            0.23716612195992776,
        ],
        [
            0.14440180508574119,
            0.2608951400467466,
            0.31593772876546944,
            0.27876532610204263,
        ],
        [
            0.30412905259317613,
            0.15162041365131648,
            0.2529224413841839,
            0.2913280923713236,
        ],
    ]
)


# translation/transversion rate
# Transitions are interchanges of A and G, or C and T
# Transversions are remianing possible interchanges
# Assumes that transitions are twice as likely as transversions for all models
kappa = 2.0 * np.ones(n_states)

routine_dict = single_decoding_routine(
    tree,
    n_nucleotids,
    A,
    b,
    n_species,
    n_states,
    pi,
    kappa,
    scaling_factors,
)
probabilities_easy = routine_dict["probabilities"]

# Plot the probabilities and store them in a file
import matplotlib.pyplot as plt

plt.plot(probabilities_easy[0, :])

# Make the y axis only 0.5 to 1
plt.ylim(0.5, 1)
plt.title("Conservation Scores")
plt.xlabel("Base Position")
plt.ylabel("Posterior Probability")
plt.savefig("probabilities.png")
