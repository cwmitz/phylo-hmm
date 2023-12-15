from copy import deepcopy

import numpy as np
from scipy.linalg import expm

from data_simulation import generate_case


# KEEP AS NUMPY ARRAYS
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
    Computes the likelihood of a given site using Felsenstein's algorithm
    with respect to a phylogenetic tree defined by Q, pi, tree.
    Parameters:
        - Q (np.matrix): the substitution rate matrix.
        - pi (np.array): the vector of background frequencies.
        - tree (dict): the tree referencing the relationships between nodes
          and the branches length.
        - sites (matrix): all the sites
    Returns:
        - (np vector): the likelihood.
    """
    likelihood_cache = {}
    total_nucleotides = 1000000
    observed_sites = sites.copy()
    nucleotide_indices = np.arange(total_nucleotides)

    def compute_likelihood(node):
        if node in likelihood_cache:
            return likelihood_cache[node]

        node_children = tree.get(node, [])
        if not node_children:
            # Node is a leaf
            leaf_likelihood = np.zeros((total_nucleotides, 4))
            leaf_likelihood[nucleotide_indices, np.floor(
                observed_sites[:, node - 1]).astype(int)] = 1
            return leaf_likelihood

        # Compute likelihood for non-leaf nodes
        left, right = node_children
        left_likelihood = compute_likelihood(left["node"])
        right_likelihood = compute_likelihood(right["node"])

        transition_matrix_left = expm(left["branch"] * Q)
        transition_matrix_right = expm(right["branch"] * Q)

        combined_left = transition_matrix_left @ left_likelihood.T
        combined_right = transition_matrix_right @ right_likelihood.T

        return (combined_left * combined_right).T

    # Compute likelihood for each node in the tree
    for node in tree:
        likelihood_cache[node] = compute_likelihood(node)

    root_likelihood = likelihood_cache[max(tree.keys())]
    return np.dot(root_likelihood, pi)


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
    m = np.max(a, axis=axis, keepdims=True)
    exp_diff = np.exp(a - m)
    sum_exp = np.sum(exp_diff, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(sum_exp), axis=axis)


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
    forward_log_prob = np.zeros((n_states, n_sites))
    backward_log_prob = np.zeros((n_states, n_sites))

    # FORWARD PROCEDURE
    forward_log_prob[:, 0] = np.log(b) + np.log(E[:, 0])
    for t in range(1, n_sites):
        for s in range(n_states):
            prob = np.log(A[:, s]) + forward_log_prob[:, t - 1]
            forward_log_prob[s, t] = np.log(E[s, t]) + sum_log(prob)

    # BACKWARD PROCEDURE
    backward_log_prob[:, -1] = 0  # log(1) = 0
    for t in range(n_sites - 2, -1, -1):
        for s in range(n_states):
            prob = np.log(A[s, :]) + np.log(E[:, t + 1]) + \
                backward_log_prob[:, t + 1]
            backward_log_prob[s, t] = sum_log(prob)

    # POSTERIOR PROBABILITIES
    log_post_probas = forward_log_prob + backward_log_prob
    log_post_probas -= sum_log(log_post_probas, axis=0)
    post_probas = np.exp(log_post_probas)

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

    # Save generated stands in fasta file format
    nucleotide_mapping = {"0": "A", "1": "C", "2": "T", "3": "G"}
    nuc_strands = [""] * 9
    for i in range(9):
        nuc_strands[i] = "".join([nucleotide_mapping[str(x)] for x in strands[i]])
    with open("sequences.fasta", "w") as file:
        for i, sequence in enumerate(nuc_strands):
            file.write(">" + str(i + 1) + "\n" + sequence + "\n")

    # Process likelihoods with Felsenstein's algorithm
    likelihoods = np.zeros((n_states, number_of_nucleotids))

    sites = np.zeros((number_of_nucleotids, n_species))
    for i in range(n_species):
        sites[:, i] = strands[i]

    # Getting emmission probabilities for each site using Felsenstein's algorithm
    for state in range(n_states):
        likelihoods[state] = felsensteins(
            Qs[state], pi[state], trees[state], sites)

    probabilities = forward_backward(A, b, likelihoods)
    return probabilities
