import pickle
import numpy as np

import torch
from torch.autograd import Variable


def get_probability(class_id, set_size):
    """ Calculates the probability of a word occuring in some corpus the classes of which follow a log-uniform 
    (Zipfian) base distribution"""
    class_prob = (np.log(class_id + 2) - np.log(class_id + 1)) / np.log(set_size + 1)
    return class_prob


def renormalize(class_probs, rejected_id):
    """ Re-normalizes the probabilities of remaining classes within the class set after the rejection of 
    some class previously present within the set. """
    rejected_mass = class_probs[rejected_id]
    class_probs[rejected_id] = 0
    remaining_mass = 1 - rejected_mass
    updated_class_probs = {class_id: class_probs[class_id] / remaining_mass for class_id in class_probs.keys()}
    return updated_class_probs


def make_sampling_array(range_max, array_path):
    """ Creates and populates the array from which the fake labels are sampled during the NCE loss calculation."""
    # Get class probabilities
    print('Computing the Zipfian distribution probabilities for the corpus items.')
    class_probs = {class_id: get_probability(class_id, range_max) for class_id in range(range_max)}

    print('Generating and populating the sampling array. This may take a while.')
    # Generate empty array
    sampling_array = np.zeros(int(1e8))
    # Determine how frequently each index has to appear in array to match its probability
    class_counts = {class_id: int(np.round((class_probs[class_id] * 1e8))) for class_id in range(range_max)}
    assert(sum(list(class_counts.values())) == 1e8), 'Counts don\'t add up to the array size!'

    # Populate sampling array
    pos = 0
    for key, value in class_counts.items():
        while value != 0:
            sampling_array[pos] = key
            pos += 1
            value -= 1
    # Save filled array into a pickle, for subsequent reuse
    with open(array_path, 'wb') as f:
        pickle.dump((sampling_array, class_probs), f)

    return sampling_array, class_probs


def sample_values(true_classes, num_sampled, unique, no_accidental_hits, sampling_array, class_probs):
    """ Samples negative items for the calculation of the NCE loss. Operates on batches of targets. """
    # Initialize output sequences
    sampled_candidates = np.zeros(num_sampled)
    true_expected_count = np.zeros(true_classes.size())
    sampled_expected_count = np.zeros(num_sampled)

    # If the true labels should not be sampled as a noise items, add them all to the rejected list
    if no_accidental_hits:
        rejected = list()
    else:
        rejected = true_classes.tolist()
    # Assign true label probabilities
    rows, cols = true_classes.size()
    for i in range(rows):
        for j in range(cols):
            true_expected_count[i][j] = class_probs[true_classes.data[i][j]]
    # Obtain sampled items and their probabilities
    print('Sampling items and their probabilities.')
    for k in range(num_sampled):
        sampled_pos = np.random.randint(int(1e8))
        sampled_idx = sampling_array[sampled_pos]
        if unique:
            while sampled_idx in rejected:
                sampled_idx = sampling_array[np.random.randint(0, int(1e8))]
        # Append sampled candidate and its probability to the output sequences for current target
        sampled_candidates[k] = sampled_idx
        sampled_expected_count[k] = class_probs[sampled_idx]
        # Re-normalize probabilities
        if unique:
            class_probs = renormalize(class_probs, sampled_idx)

    # Process outputs before they are returned
    sampled_candidates = sampled_candidates.astype(np.int64, copy=False)
    true_expected_count = true_expected_count.astype(np.float32, copy=False)
    sampled_expected_count = sampled_expected_count.astype(np.float32, copy=False)

    return Variable(torch.LongTensor(sampled_candidates)), \
           Variable(torch.FloatTensor(true_expected_count)), \
           Variable(torch.FloatTensor(sampled_expected_count))
