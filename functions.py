import json
from copy import deepcopy

import numpy as np
from scipy.linalg import expm

from data_simulation import generate_case, rate_sub_HKY, scale_branches_length


# COPIED FROM FELSENSTEIN.PY
def np_pruning(Q, pi, tree, sites):
    """
    Computes the likelihood at a given site, for a given instance of this
    site with respect to a phylogenetic tree defined by Q, pi, tree, using
    Felsenstein's pruning algorithm.
        Args:
            - Q (np.matrix): the subsitution rate matrix.
            - pi (np.array): the vector of background frequencies.
            - tree (dict): the tree referencing the relationships between nodes
            and the branches length.
            - site (np matrix): all the sites
        Output:
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
    :param a: vector or matrix
    :param axis: axis along which the sum is performed, useful if 'a' is a
    matrix
    :return: m + log(sum_i exp(a_i - m)) with m = max(a)
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
    Sum-Product
    algorithm or Viterbi. Logarithms are used to avoid numerical errors.
    :param A: matrix of state-transition probabilities (n_states rows, n_states columns)
    :param b: vector of initial-state probabilities (dimension n_states)
    :param E: matrix of emission probabilities computed with Felstenstein
    algorithm (n_states rows, n_sites columns)
    :param mode: string to precise how to compute the log alpha-messages. Must
    be 'max' for Viterbi,
            'sum' for Sum-Product
    :return: if 'max', matrix of log alpha-messages and the argmax matrix ; if
    'sum', matrix of log alpha-messages
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
    :param A: matrix of state-transition probabilities (n_states rows, n_states columns)
    :param E: matrix of emission probabilities computed with Felstenstein
    algorithm (n_states rows, n_sites columns)
    :return: matrix of log beta-messages
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
    :param A: matrix of state-transition probabilities (n_states rows, n_states columns)
    :param b: vector of initial-state probabilities (dimension n_states)
    :param E: matrix of emission probabilities computed with Felstenstein
    algorithm (n_states rows, n_sites columns)
    :return: matrix of posterior probabilities
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


def single_decoding_routine(
    tree,
    number_of_nucleotids,
    A,
    b,
    n_species,
    pi,
    kappa,
    scaling_factors,
):
    """Generates a sequence of states and an associated list of strands b
    based on parameters. Then decodes those using a phylogenetic HMM model.
    """
    nbState = A.shape[0]

    # Scale true topology tree by scaling factor associated with each state
    trees = []
    for j in range(nbState):
        trees.append(scale_branches_length(tree, scale=scaling_factors[j]))

    strands, states = generate_case(A, b, pi, kappa, trees, number_of_nucleotids)

    # Process likelihoods with Felsenstein's algorithm

    Qs = rate_sub_HKY(pi, kappa)
    # Process likelihoods with Felsenstein's algorithm

    likelihoods = np.zeros((nbState, number_of_nucleotids))
    sites = np.zeros((number_of_nucleotids, n_species))
    for i in range(n_species):
        sites[:, i] = strands[i]
    for state in range(nbState):
        tree = trees[state]
        Q = Qs[state]
        p = pi[state]
        likelihoods[state] = np_pruning(Q, p, tree, sites)

    # VITERBI PARAMETERS
    S = range(nbState)

    probabilities = forward_backward(A, b, likelihoods)
    return {
        "real_states": states,
        "probabilities": probabilities,
        "decoded_states": np.argmax(probabilities, axis=0),
    }
