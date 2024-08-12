import math
import scipy
from tqdm import tqdm
import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from pactl.data import WrapperDataset


class CanariesDataset(WrapperDataset):
    def __init__(self, dataset, n_labels=10, num_canaries=0):
        super().__init__(dataset)

        self.C = n_labels
        # self.data = dataset.data

        if num_canaries > 0:
            labels = np.array(self.targets)
            mask = np.arange(0, len(labels)) < num_canaries
            np.random.seed(1024)
            np.random.shuffle(mask)
            rnd_labels = np.random.choice(self.C, mask.sum())
            labels[mask] = rnd_labels
            # we need to explicitly cast the labels from npy.int64 to
            # builtin int type, otherwise pytorch will fail...
            labels = [int(x) for x in labels]
            self.noisy_targets = labels

    def __getitem__(self, i):
        X, y = super().__getitem__(i)
        y = self.noisy_targets[i]
        return X, y


def insert_canaries(train_data, num_classes, audit_size, seed, label_noise=1):
    np.random.seed(seed)
    audit_index = np.random.choice(list(range(0, len(train_data))), size=audit_size, replace=False)  # m
    non_sampled_index = list(set(list(range(0, len(train_data)))) - set(audit_index))  # n-m
    # flip bernoulli coin
    bernoulli_index = np.random.binomial(n=1, p=0.5, size=len(audit_index))
    non_mem_index = audit_index[np.where(bernoulli_index == 0)]
    mem_index = audit_index[np.where(bernoulli_index == 1)]
    new_train_index = list(mem_index) + non_sampled_index
    # split data
    new_train_data = Subset(train_data, new_train_index)
    new_train_data.num_classes = num_classes
    mem_data = Subset(train_data, mem_index)
    mem_data.data = train_data.data[list(mem_index)]
    mem_data.targets = list(np.array(train_data.targets)[list(mem_index)])
    non_mem_data = Subset(train_data, non_mem_index)
    non_mem_data.data = train_data.data[list(non_mem_index)]
    non_mem_data.targets = list(np.array(train_data.targets)[list(non_mem_index)])
    # insert canary
    num_canaries_mem = math.floor(len(mem_data) * label_noise)
    mem_data_canary = CanariesDataset(mem_data, n_labels=num_classes, num_canaries=num_canaries_mem)
    num_canaries_non_mem = math.floor(len(non_mem_data) * label_noise)
    non_mem_data_canary = CanariesDataset(non_mem_data, n_labels=num_classes, num_canaries=num_canaries_non_mem)

    return mem_data_canary, non_mem_data_canary, new_train_data


# deprecated
def generate_auditing_data(train_data, audit_size):
    np.random.seed(1024)
    audit_index = np.random.choice(list(range(0, len(train_data))), size=audit_size, replace=False)
    non_sampled_index = list(set(list(range(0, len(train_data)))) - set(audit_index))
    bernoulli_index = np.random.binomial(n=1, p=0.5, size=len(audit_index))
    non_mem_index = audit_index[np.where(bernoulli_index == 1)]
    # mem_index = list(audit_index[np.where(bernoulli_index == 0)]) + non_sampled_index
    mem_index = list(audit_index[np.where(bernoulli_index == 0)])
    member_data = Subset(train_data, mem_index)
    non_mem_data = Subset(train_data, non_mem_index)
    return member_data, non_mem_data


def p_value_DP_audit(m, r, v, eps, delta=0):
    """
    Args:
        m = number of examples, each included independently with probability 0.5
        r = number of guesses (i.e. excluding abstentions)
        v = number of correct guesses by auditor
        eps,delta = DP guarantee of null hypothesis
    Returns:
        p-value = probability of >=v correct guesses under null hypothesis
    """
    assert 0 <= v <= r <= m
    assert eps >= 0
    assert 0 <= delta <= 1
    q = 1 / (1 + math.exp(-eps))  # accuracy of eps-DP randomized response
    beta = scipy.stats.binom.sf(v - 1, r, q)  # = P[Binomial(r, q) >= v]
    alpha = 0
    sum = 0  # = P[v > Binomial(r, q) >= v - i]
    for i in range(1, v + 1):
        sum = sum + scipy.stats.binom.pmf(v - i, r, q)
        if sum > i * alpha:
            alpha = sum / i
    p = beta  # + alpha * delta * 2 * m
    # print("p", p)
    return min(p, 1)


def get_eps_audit(m, r, v, p, delta):
    """
    Args:
        m = number of examples, each included independently with probability 0.5
        r = number of guesses (i.e. excluding abstentions)
        v = number of correct guesses by auditor
        p = 1-confidence e.g. p=0.05 corresponds to 95%
    Returns:
        lower bound on eps i.e. algorithm is not (eps,delta)-DP
    """
    assert 0 <= v <= r <= m
    assert 0 <= delta <= 1
    assert 0 < p <= 1
    eps_min = 0  # maintain p_value_DP(eps_min) < p
    eps_max = 1  # maintain p_value_DP(eps_max) >= p

    while p_value_DP_audit(m, r, v, eps_max, delta) < p: eps_max = eps_max + 1
    for _ in range(30):  # binary search
        eps = (eps_min + eps_max) / 2
        if p_value_DP_audit(m, r, v, eps, delta) < p:
            eps_min = eps
        else:
            eps_max = eps
    return eps_min


def find_O1_pred_quick(member_loss_values, non_member_loss_values, delta=0.):
    # Create labels for real and generated loss values
    member_labels = np.ones_like(member_loss_values)
    non_member_labels = np.ones_like(non_member_loss_values) * -1

    # Concatenate loss values and labels
    all_losses = np.concatenate((member_loss_values, non_member_loss_values))
    all_labels = np.concatenate((member_labels, non_member_labels))

    # sort loss values
    ind = np.argsort(all_losses)  # ascending order
    sorted_losses = all_losses[ind]
    sorted_labels = all_labels[ind]

    # get max pred acc
    threshold_pos = np.arange(5, len(ind)-5+1, 5)
    # 5: to make sure no edge case of very little predictions gets accidently high acc
    # 1: increase gap if slow
    best_acc = 0
    best_eps = -1
    best_num_guesses = 0
    best_correct_guesses = 0
    best_t_neg, best_t_pos = -1, -1
    for t_neg in threshold_pos:
        for t_pos in threshold_pos:
            total_pred = t_neg + t_pos
            if total_pred > len(all_labels):
                break
            guesses = np.zeros(len(all_labels))
            guesses[:t_neg] = -1
            guesses[-t_pos:] = 1
            correct_pred = (guesses == sorted_labels).sum()
            acc = correct_pred / total_pred
            eps = get_eps_audit(len(all_labels), best_num_guesses, best_correct_guesses, p=0.05, delta=0)

            # if acc > best_acc:
            if eps > best_eps:
                best_eps = eps
                best_acc = acc
                best_num_guesses = total_pred
                best_correct_guesses = correct_pred
                best_t_neg = t_neg
                best_t_pos = t_pos

    # eps = get_eps_audit(len(all_labels), best_num_guesses, best_correct_guesses, p=0.05, delta=0)

    metric = {
        'audit_eps': best_eps, 'threshold_t_neg': best_t_neg, 'threshold_t_pos': best_t_pos,
        'best_accuracy': best_acc,
        'total_predictions': best_num_guesses, 'correct_predictions': best_correct_guesses
    }

    return metric


# def find_O1_pred_quick(member_loss_values, non_member_loss_values, delta=0.):
#     """
#     Args:
#         member_loss_values: NumPy array containing member loss values
#         non_member_loss_values: NumPy array containing non_member loss values
#     Returns:
#      best_eps: largest audit (epsilon) value that can be returned for a particular p value
#     """
#
#     # Create labels for real and generated loss values
#     member_labels = np.ones_like(member_loss_values)
#     non_member_labels = np.zeros_like(non_member_loss_values)
#
#     # Concatenate loss values and labels
#     all_losses = np.concatenate((member_loss_values, non_member_loss_values))
#     all_labels = np.concatenate((member_labels, non_member_labels))
#
#     # Step 1: Find t_pos that maximizes precision for positive predictions
#     best_precision = 0
#     best_t_pos = 0
#     # threshold_range = np.arange(np.min(all_losses), np.max(all_losses) + 0.01, 0.01)
#     threshold_range = np.arange(np.min(all_losses), np.max(all_losses) + 0.01, 0.05)
#     # use fixed length threshold
#     # threshold_range = np.linspace(start=np.min(all_losses), stop=np.max(all_losses) + 0.01, num=20)
#     results, recall = [], []
#     best_accuracy = 0
#     best_t_neg = 0
#     total_predictions = 0
#     correct_predictions = 0
#     best_eps = 0
#     p = 0.05
#     tmp_best_tp = 0
#     for t_pos in tqdm(threshold_range):
#         positive_predictions = all_losses[all_losses <= t_pos]
#         if len(positive_predictions) == 0:
#             continue
#
#         true_positives = np.sum(all_labels[all_losses <= t_pos] == 1)
#
#         # eps = get_eps_audit(len(all_labels), len(positive_predictions), true_positives, p, delta)
#         # # precision = true_positives / len(positive_predictions)
#         # if eps > best_eps:
#         #     print("EPSILON UPDATE:", eps)
#         #     best_eps = eps
#         #     best_t_pos = t_pos
#         # # recalls = true_positives / np.sum(all_labels == 1)
#         # # recall.append(recalls)
#         if true_positives > tmp_best_tp:
#             tmp_best_tp = true_positives
#             best_t_pos = t_pos
#
#     # unnest inside loop for faster computation
#     # Step 2: With t_pos fixed, find t_neg that maximizes overall accuracy
#     for t_neg in tqdm(reversed(threshold_range)):
#         if t_neg <= best_t_pos:
#             break
#         confident_predictions = all_losses[(all_losses <= best_t_pos) | (all_losses >= t_neg)]
#         r = len(confident_predictions)
#         mask_pos = (confident_predictions <= best_t_pos) & (
#                     all_labels[(all_losses <= best_t_pos) | (all_losses >= t_neg)] == 1)
#         mask_neg = (confident_predictions >= t_neg) & (
#                     all_labels[(all_losses <= best_t_pos) | (all_losses >= t_neg)] == 0)
#
#         v = np.sum(np.logical_or(mask_pos, mask_neg))
#
#         if r > 0:
#             accuracy = v / r
#             # eps = get_eps_audit(len(all_labels), r, v, p, delta)
#             # if eps > best_eps:
#             #     best_eps = eps
#             #     best_t_neg = t_neg
#             #     total_predictions = r
#             #     correct_predictions = v
#             if accuracy > best_accuracy:
#                 best_accuracy = accuracy
#                 best_t_neg = t_neg
#                 total_predictions = r
#                 correct_predictions = v
#
#     best_eps = get_eps_audit(len(all_labels), total_predictions, correct_predictions, p, delta)
#
#     metric = {
#         'best_eps': best_eps, 'threshold_t_neg': best_t_neg, 'threshold_t_pos': best_t_pos,
#         'best_precision': best_precision, 'best_accuracy': best_accuracy,
#         'total_predictions': total_predictions, 'correct_predictions': correct_predictions
#     }
#     # print(f"Best eps: {best_eps} with thresholds (t_neg, t_pos): ({best_t_neg}, {best_t_pos})")
#     # print(f"Best precision for t_pos: {best_precision} with t_pos: {best_t_pos}")
#     # print(f"Best accuracy: {best_accuracy} with thresholds (t_neg, t_pos): ({best_t_neg}, {best_t_pos})")
#
#     # # Save results to CSV file
#     # output_csv_path = "swiss_audit_over.csv"
#     # with open(output_csv_path, 'w', newline='') as csvfile:
#     #     fieldnames = ['t_pos', 't_neg', 'best_precision', 'best_accuracy', 'recall', 'total_predictions',
#     #                   'correct_predictions']
#     #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     #
#     #     writer.writeheader()
#     #     for result in results:
#     #         writer.writerow(result)
#
#     # return total_predictions, correct_predictions, len(all_losses), metric
#     return metric


def find_O1_pred(member_loss_values, non_member_loss_values, delta=0.):
    """
    Args:
        member_loss_values: NumPy array containing member loss values
        non_member_loss_values: NumPy array containing non_member loss values
    Returns:
     best_eps: largest audit (epsilon) value that can be returned for a particular p value
    """

    # Create labels for real and generated loss values
    member_labels = np.ones_like(member_loss_values)
    non_member_labels = np.zeros_like(non_member_loss_values)

    # Concatenate loss values and labels
    all_losses = np.concatenate((member_loss_values, non_member_loss_values))
    all_labels = np.concatenate((member_labels, non_member_labels))

    # Step 1: Find t_pos that maximizes precision for positive predictions
    best_precision = 0
    best_t_pos = 0
    # threshold_range = np.arange(np.min(all_losses), np.max(all_losses) + 0.01, 0.01)
    threshold_range = np.arange(np.min(all_losses), np.max(all_losses) + 0.01, 0.1)
    results, recall = [], []
    best_accuracy = 0
    best_t_neg = 0
    total_predictions = 0
    correct_predictions = 0
    best_eps = 0
    p = 0.05
    for t_pos in tqdm(threshold_range):
        positive_predictions = all_losses[all_losses <= t_pos]
        if len(positive_predictions) == 0:
            continue

        true_positives = np.sum(all_labels[all_losses <= t_pos] == 1)

        eps = get_eps_audit(len(all_labels), len(positive_predictions), true_positives, p, delta)
        # precision = true_positives / len(positive_predictions)
        if eps > best_eps:
            print("EPSILON UPDATE:", eps)
            best_eps = eps
            best_t_pos = t_pos
        # recalls = true_positives / np.sum(all_labels == 1)
        # recall.append(recalls)

    # unnest inside loop for faster computation
    # Step 2: With t_pos fixed, find t_neg that maximizes overall accuracy
    for t_neg in tqdm(reversed(threshold_range)):
        if t_neg <= best_t_pos:
            break
        confident_predictions = all_losses[(all_losses <= best_t_pos) | (all_losses >= t_neg)]
        r = len(confident_predictions)
        mask_pos = (confident_predictions <= best_t_pos) & (
                    all_labels[(all_losses <= best_t_pos) | (all_losses >= t_neg)] == 1)
        mask_neg = (confident_predictions >= t_neg) & (
                    all_labels[(all_losses <= best_t_pos) | (all_losses >= t_neg)] == 0)

        v = np.sum(np.logical_or(mask_pos, mask_neg))

        if r > 0:
            accuracy = v / r
            eps = get_eps_audit(len(all_labels), r, v, p, delta)
            if eps > best_eps:
                best_eps = eps
                best_t_neg = t_neg
                total_predictions = r
                correct_predictions = v
            if accuracy > best_accuracy:
                best_accuracy = accuracy

        # results.append({
        #     't_pos': t_pos,
        #     't_neg': best_t_neg,
        #     'best_precision': precision,
        #     'best_accuracy': best_accuracy,
        #     'recall': recall,
        #     'total_predictions': r,
        #     'correct_predictions': v
        # })  # suppress logging

    metric = {
        'best_eps': best_eps, 'threshold_t_neg': best_t_neg, 'threshold_t_pos': best_t_pos,
        'best_precision': best_precision, 'best_accuracy': best_accuracy,
        'total_predictions': total_predictions, 'correct_predictions': correct_predictions
    }
    print(f"Best eps: {best_eps} with thresholds (t_neg, t_pos): ({best_t_neg}, {best_t_pos})")
    print(f"Best precision for t_pos: {best_precision} with t_pos: {best_t_pos}")
    print(f"Best accuracy: {best_accuracy} with thresholds (t_neg, t_pos): ({best_t_neg}, {best_t_pos})")

    # # Save results to CSV file
    # output_csv_path = "swiss_audit_over.csv"
    # with open(output_csv_path, 'w', newline='') as csvfile:
    #     fieldnames = ['t_pos', 't_neg', 'best_precision', 'best_accuracy', 'recall', 'total_predictions',
    #                   'correct_predictions']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #
    #     writer.writeheader()
    #     for result in results:
    #         writer.writerow(result)

    # return total_predictions, correct_predictions, len(all_losses), metric
    return metric
