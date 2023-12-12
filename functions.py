import json
from copy import deepcopy

import numpy as np
from scipy.linalg import expm

from data_simulation import generate_case


# COPIED FROM FELSENSTEIN.PY
def np_pruning(Q, pi, tree, sites):
    """
    Computes the likelihood at a given site, for a given instance of this
    site with respect to a phylogenetic tree defined by Q, pi, tree, using
    Felsenstein's pruning algorithm.
        Parameters:
            - Q (np.matrix): the subsitution rate matrix.
            - pi (np.array): the vector of background frequencies.
            - tree (dict): the tree referencing the relationships between nodes
            and the branches length.
            - site (np matrix): all the sites
        Returns:
            - (np vector): the likelihood.
    """
    dynamic_probas = {}  # this dictionary will allow to keep records of
    # posterior probabilities which were already calculated.

    nb_nucleotides, n_species = sites.shape
    observation = sites.copy()
    x_index = np.arange(nb_nucleotides)  # frequently used variable

    def posterior_proba(node):
        if node in dynamic_probas:
            return dynamic_probas[node]
        else:
            childs = tree[node]
            if childs:
                left_child = childs[0]["node"]
                right_child = childs[1]["node"]
                post_proba_left = posterior_proba(left_child)
                post_proba_right = posterior_proba(right_child)
                prob_matrix_left = expm(childs[0]["branch"] * Q)
                prob_matrix_right = expm(childs[1]["branch"] * Q)
                # import pdb; pdb.set_trace()
                left_likelihood = prob_matrix_left.dot(post_proba_left.T)
                right_likelihood = prob_matrix_right.dot(post_proba_right.T)
                return (left_likelihood * right_likelihood).T
            else:
                # in this case the node is a leaf
                likelihood = np.zeros((nb_nucleotides, 4))
                likelihood[
                    x_index, np.floor(observation[:, node - 1]).astype("int")
                ] = 1
                return likelihood

    for node in tree:
        dynamic_probas[node] = posterior_proba(node)

    return dynamic_probas[max(tree.keys())].dot(pi)


# COPIED FROM VITERBI_SUMPRODUCT.PY
def sum_log(a, axis=0):
    """
    Sum when working with logarithms to avoid numerical errors
        Parameters:
            - a (np.array): vector or matrix
            - axis (int): axis along which the sum is performed, useful if 'a' is a
            matrix
        Returns:
            - m + log(sum_i exp(a_i - m)) with m = max(a)
    """
    # 'a' is a vector
    if a.ndim == 1:
        m = max(a)
        return m + np.log(sum(np.exp(a - m)))
    # 'a' is a matrix
    else:
        m = np.max(a, axis=axis)
        diff = a - m[:, np.newaxis] if axis == 1 else a - m
        return m + np.log(np.sum(np.exp(diff), axis=axis))


# COPIED FROM VITERBI_SUMPRODUCT.PY
def forward(A, b, E, mode):
    """
    Forward pass: computes the logarithms of alpha-messages associated to the
    Sum-Product algorithm or Viterbi. Logarithms are used to avoid numerical errors.
        Parameters:
            - A (np.array): matrix of state-transition probabilities (n_states rows, n_states columns)
            - b (np.array): vector of initial-state probabilities (dimension n_states)
            - E (np.array): matrix of emission probabilities computed with Felstenstein algorithm (n_states rows, n_sites columns)
            - mode (str): string to precise how to compute the log alpha-messages. Must be 'max' for Viterbi, 'sum' for Sum-Product
        Returns:
            - if 'max', matrix of log alpha-messages and the argmax matrix ; if 'sum', matrix of log alpha-messages
    """
    if mode != "max" and mode != "sum":
        return "Error: Input parameter mode must be 'sum' or 'max'!"
    else:
        # Initialization
        n_states, n_sites = E.shape
        alpha_log = np.zeros((n_states, n_sites))
        alpha_log[:, 0] = np.log(b) + np.log(E[:, 0])
        alpha_argmax = np.zeros(
            (n_states, n_sites), dtype=int
        )  # useful for mode 'max' only
        # Recursion
        for t in range(1, n_sites):
            for s in range(n_states):
                prob = np.log(A[:, s]) + alpha_log[:, t - 1]
                if mode == "sum":
                    prob = sum_log(prob)
                else:
                    alpha_argmax[s, t] = np.argmax(prob)
                    prob = max(prob)
                alpha_log[s, t] = np.log(E[s, t]) + prob
        return alpha_log if mode == "sum" else [alpha_log, alpha_argmax]


# COPIED FROM VITERBI_SUMPRODUCT.PY
def backward(A, E):
    """
    Backward pass: computes the logarithms of beta-messages for the Sum-Product
    algorithm
        Parameters:
            - A (np.array): matrix of state-transition probabilities (n_states rows, n_states columns)
            - E (np.array): matrix of emission probabilities computed with Felstenstein algorithm (n_states rows, n_sites columns)
        Returns:
            - matrix of log beta-messages
    """
    # Initialization
    n_states, n_sites = E.shape
    beta_log = np.zeros((n_states, n_sites))
    # Recursion
    for t in range(n_sites - 2, -1, -1):
        beta_log[:, t] = sum_log(
            np.log(A) + np.log(E[:, t + 1]) + beta_log[:, t + 1], axis=1
        )
    return beta_log


# COPIED FROM VITERBI_SUMPRODUCT.PY
def forward_backward(A, b, E):
    """
    Sum-Product algorithm (forward-backward procedure)
        Parameters:
            - A (np.array): matrix of state-transition probabilities (n_states rows, n_states columns)
            - b (np.array): vector of initial-state probabilities (dimension n_states)
            - E (np.array): matrix of emission probabilities computed with Felstenstein algorithm (n_states rows, n_sites columns)
        Returns:
            - matrix of posterior probabilities
    """
    n_states, n_sites = E.shape
    post_probas = np.zeros((n_states, n_sites))
    # Forward and backward procedure to compute the logarithms of alpha and
    # beta messages
    alpha_log = forward(A, b, E, "sum")
    beta_log = backward(A, E)
    # Posterior probabilities computation using the log method
    post_probas = np.exp(alpha_log + beta_log - sum_log(alpha_log + beta_log, axis=0))
    return post_probas


def rate_sub_HKY(pi, kappa, n_states):
    """
    Define the rate substitution matrices according to the HKY model for
    all states
        Parameters:
            - pi (nparray : nbState, alphabetSize) nucleotids background
            frequencies, state dependent
           - kappa (np vector, size nb states) translation/transversion rate
    returns : Q (np array, nb states x alphabetSize x alphabetSize)
    the rate substition matrices for each state
    """
    alphabetSize = 4
    Q = np.zeros((n_states, alphabetSize, alphabetSize))
    for j in range(n_states):
        for i in range(alphabetSize):
            Q[j, i, :] = pi[j]
            Q[j, i, (i + 2) % alphabetSize] *= kappa[j]
            # put in each diagonal a term such that the rows sums to 0
            Q[j, i, i] -= np.sum(Q[j, i, :])
    return Q


def single_decoding_routine(
    tree,
    number_of_nucleotids,
    A,
    b,
    n_species,
    n_states,
    pi,
    kappa,
    scaling_factors,
):
    """
    Generates a sequence of states and an associated list of strands b
    based on parameters. Then decodes those using a phylogenetic HMM model.

        Parameters:
            - tree: Dictionary of true tree topology with branch lengths
            - number_of_nucleotids: 1,000,000
            - A: Matrix of state-transition probabilities
            - b: Vector of initial-state probabilities
            - n_species: Number of species
            - n_states: Number of states
            - pi: Matrix of nucleotid background frequencies
            - kappa: Vector of translation/transversion rates
            - scaling_factors: Vector of scaling factors for each state
    """
    # Scale true topology tree by scaling factor associated with each state
    trees = []
    for j in range(n_states):
        tree_copy = deepcopy(tree)
        max_node = max(tree_copy.keys())

        def rescale_node(node):
            children = tree_copy[node]
            if children:
                for child in children:
                    new_child = child["node"]
                    rescale_node(new_child)
                    child["branch"] *= scaling_factors[j]

        rescale_node(max_node)

        trees.append(tree_copy)

    # Define rate matrix Q for each state based on on the HKY model which implies Q has form
    # corresponding to (1) on page 4 of base paper also detailed on page 421 of:
    # Combining Phylogenetic and Hidden Markov Models in Biosequence Analysis
    # Adam Siepel and David Haussler
    # Journal of Computational Biology 2004 11:2-3, 413-428
    Qs = rate_sub_HKY(pi, kappa, n_states)

    # Generate strands and states
    strands, states = generate_case(Qs, A, b, pi, trees, number_of_nucleotids)

    # FUNCTION TO CONVERT LIST OF FORM [0,1,2,3, ...] TO STRING OF FORM AGCT...
    # WHERE 0 CORRESPONDS TO "A", 1 CORRESPONDS TO "C", 2 CORRESPONDS TO "T", AND 3 CORRESPONDS TO "G"
    # - strands (list of np vector uint8: number_of_nucleotids)
    # list of the sequence of nucleotids for each taxa
    nucleotide_mapping = {"0": "A", "1": "C", "2": "T", "3": "G"}
    nuc_strands = [""] * 9
    for i in range(9):
        nuc_strands[i] = "".join([nucleotide_mapping[str(x)] for x in strands[i]])

    # save nuc_strings into a file of fasta format
    with open("sequences.fasta", "w") as file:
        for i, sequence in enumerate(nuc_strands):
            file.write(">" + str(i + 1) + sequence + "\n")

    # Process likelihoods with Felsenstein's algorithm

    likelihoods = np.zeros((n_states, number_of_nucleotids))
    sites = np.zeros((number_of_nucleotids, n_species))
    for i in range(n_species):
        sites[:, i] = strands[i]
    for state in range(n_states):
        tree = trees[state]
        Q = Qs[state]
        p = pi[state]
        likelihoods[state] = np_pruning(Q, p, tree, sites)

    # VITERBI PARAMETERS
    S = range(n_states)

    probabilities = forward_backward(A, b, likelihoods)
    return {
        "real_states": states,
        "probabilities": probabilities,
        "decoded_states": np.argmax(probabilities, axis=0),
    }
