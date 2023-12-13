import json
from copy import deepcopy

import numpy as np
from scipy.linalg import expm

from data_simulation import generate_case


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
            Q[j, i, (i + 2) % alphabetSize] *= kappa
            # put in each diagonal a term such that the rows sums to 0
            Q[j, i, i] -= np.sum(Q[j, i, :])
    return Q


def felsensteins(Q, pi, tree, sites):
    """
    Uses Felsenstein's algorithm to compute the likelihood of a given site
    for a given instance of this site with respect to a phylogenetic tree
    defined by Q, pi, tree.
        Parameters:
            - Q (np.matrix): the subsitution rate matrix.
            - pi (np.array): the vector of background frequencies.
            - tree (dict): the tree referencing the relationships between nodes
            and the branches length.
            - site (matrix): all the sites
        Returns:
            - (np vector): the likelihood.
    """
    emmission_probs = {}

    nb_nucleotides = 1000000
    observation = sites.copy()
    x_index = np.arange(nb_nucleotides)  # frequently used variable

    def posterior_proba(node):
        if node in emmission_probs:
            return emmission_probs[node]
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
        emmission_probs[node] = posterior_proba(node)

    return emmission_probs[max(tree.keys())].dot(pi)


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
    if a.ndim == 1:
        m = max(a)
        return m + np.log(sum(np.exp(a - m)))
    else:
        m = np.max(a, axis=axis)
        diff = a - m[:, np.newaxis] if axis == 1 else a - m
        return m + np.log(np.sum(np.exp(diff), axis=axis))


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
    # FORWARD PROCEDURE
    # Initialization
    forward_log_prob = np.zeros((n_states, n_sites))
    forward_log_prob[:, 0] = np.log(b) + np.log(E[:, 0])
    # Recursion
    for t in range(1, n_sites):
        for s in range(n_states):
            prob = np.log(A[:, s]) + forward_log_prob[:, t - 1]
            prob = sum_log(prob)
            forward_log_prob[s, t] = np.log(E[s, t]) + prob

    # BACKWARD PROCUEDURE
    backward_log_prob = np.zeros((10, 1000000))
    for t in range(n_sites - 2, -1, -1):
        backward_log_prob[:, t] = sum_log(
            np.log(A) + np.log(E[:, t + 1]) + backward_log_prob[:, t + 1], axis=1
        )

    # Posterior probabilities computation using the log method
    post_probas = np.exp(
        forward_log_prob
        + backward_log_prob
        - sum_log(forward_log_prob + backward_log_prob, axis=0)
    )
    return post_probas


def get_probabilities(
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
    strands, _ = generate_case(Qs, A, b, pi, trees, number_of_nucleotids)

    # Process likelihoods with Felsenstein's algorithm
    likelihoods = np.zeros((n_states, number_of_nucleotids))

    sites = np.zeros((number_of_nucleotids, n_species))
    for i in range(n_species):
        sites[:, i] = strands[i]

    # Getting emmission probabilities for each site using Felsenstein's algorithm
    for state in range(n_states):
        likelihoods[state] = felsensteins(Qs[state], pi[state], trees[state], sites)

    probabilities = forward_backward(A, b, likelihoods)
    return probabilities
