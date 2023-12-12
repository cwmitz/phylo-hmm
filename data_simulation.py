import argparse
from copy import deepcopy

import numpy as np
from scipy.linalg import expm


def generate_case(Qs, A, b, pi, trees, number_of_nucleotids):
    """
    Generate a test case with DNA strands and the list of Ground truth states
        Parameters:
            - Qs: List of substitution rate matrices
            - A: Matrix of state-transition probabilities
            - b: Vector of initial-state probabilities
            - pi: Matrix of nucleotid background frequencies
            - kappa: Vector of translation/transversion rates
            - trees: List of scaled phylogenetic trees
            - number_of_nucleotids: Number of nucleotid in each strand, 1,000,000
        Returns:
            - strands (list of np vector uint8: number_of_nucleotids)
               list of the sequence of nucleotids for each taxa
            - states (np vector uint8: number_of_nucleotids) list of ground
               truth states for each size
    """
    # GT states
    states = generate_gt_state(A, b, number_of_nucleotids)
    # initial values
    X = generate_initial_vector(pi, states)

    strands = evolution(X, states, trees, Qs)

    return strands, states


def generate_initial_vector(pi, states):
    """Return a random vector of nucleotids as integers
    Args:
        - pi (nparray : nbState, alphabetSize) nucleotids background
        frequencies, state dependent
        - states (nparray: nbNucleotids), vector of the state for each site
    Output:
        - np.vector with values between 0 and size(b)-1
        follows distribution b
    """
    nbState, alphabetSize = pi.shape
    nbNucleotids = states.shape[0]

    cumsum = np.cumsum(pi, axis=1)
    random_values = np.random.rand(nbNucleotids)
    X = np.zeros(nbNucleotids, dtype=np.uint8)
    # now let us draw according to a discrete law in a vectorial way
    for i in range(alphabetSize):
        X[random_values < cumsum[states, i]] = i
        # we erase values that are lower than cumsum[i] to prevent the
        # corresponding nucleotid to be overwritten at the following step
        random_values[random_values < cumsum[states, i]] = 1

    return X


def generate_gt_state(A, b, nbNucleotids):
    """Use the state transition matrix A to generate of state path
    Args:
        - A (np matrix) state transition matrix
        - b (array of float, sums to 1) initial discrete distribution of
        states
        - nbNucleotids (int) length of the DNA in Nucleotids
    Output:
        - np. vector of int from 0 to nbState-1
    """
    states = np.empty(nbNucleotids, dtype=np.uint8)

    nbState = A.shape[0]
    # the first one is drawn with the law b
    discrete_law = np.cumsum(b)

    x = np.random.rand(1)[0]

    index = 0
    while x > discrete_law[index]:
        index += 1
    states[0] = index
    for i in range(nbNucleotids - 1):
        # draw the next state using the state transition matrix
        discrete_law = np.cumsum(A[states[i]])

        x = np.random.rand(1)[0]

        index = 0
        while x > discrete_law[index]:
            index += 1
        states[i + 1] = index
    return states


def evolution(X, states, trees, Q):
    """Use a vector of DNA X and make it evolve by running it through a
    phylogenetic tree Q
        Args:
            - X (np vector):  nucleotids as integers
            - trees (dict): list of trees as dictionnaries
            - states (np vector): state path
            - Q (narray ): list of substitution rate matrix
        Output:
            - tree with same shape but randomised branches length
    """
    nbState = Q.shape[0]
    alphabetSize = Q.shape[1]

    def evolve(node, strand):
        children = trees[0][node]
        if children:
            res = []
            for c in range(len(children)):
                new_Q = np.zeros_like(Q)
                # compute probability matrices for every state for l&r branches
                for j in range(nbState):
                    new_br = trees[j][node][c]["branch"]
                    new_Q[j] = expm(new_br * Q[j])

                new_strand = np.zeros_like(strand)
                # the new strand is drown randomly from the previous one
                # using the probability matrix
                cumsum = np.cumsum(new_Q, axis=2)
                random_values = np.random.rand(strand.shape[0])
                # vectorial discrete draw
                for i in range(alphabetSize):
                    new_strand[random_values < cumsum[states, strand, i]] = i
                    random_values[random_values < cumsum[states, strand, i]] = 1

                new_child = children[c]["node"]
                res += evolve(new_child, new_strand)
            return res
        else:
            return [strand]

    return evolve(max(trees[0].keys()), X)


def parse_args():
    parser = argparse.ArgumentParser("Data generation script")

    parser.add_argument("tree_path", help="path of a JSON file encoding a tree")
    parser.add_argument(
        "-n",
        "--number_of_nucleotids",
        type=int,
        help="number of genrated nucleotids for each taxa",
    )
    return parser.parse_args()
