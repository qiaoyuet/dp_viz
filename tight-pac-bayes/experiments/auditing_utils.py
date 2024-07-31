import numpy as np
import math
import scipy
from tqdm import tqdm
from torch.utils.data import Subset


def generate_auditing_data(train_data, audit_size):
    np.random.seed(1024)
    audit_index = np.random.choice(list(range(0, len(train_data))), size=audit_size, replace=False)
    non_sampled_index = list(set(list(range(0, len(train_data)))) - set(audit_index))
    bernoulli_index = np.random.binomial(n=1, p=0.5, size=len(audit_index))
    non_mem_index = audit_index[np.where(bernoulli_index == 1)]
    # mem_index = list(audit_index[np.where(bernoulli_index == 0)]) + non_sampled_index
    mem_index = list(audit_index[np.where(bernoulli_index == 0)])  # looser results but fast
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
    non_member_labels = np.zeros_like(non_member_loss_values)

    # Concatenate loss values and labels
    all_losses = np.concatenate((member_loss_values, non_member_loss_values))
    all_labels = np.concatenate((member_labels, non_member_labels))

    # sort loss values
    ind = np.argsort(all_losses)  # accending order
    sorted_losses = all_losses[ind]
    sorted_labels = all_labels[ind]

    correct_predictions = 0
    best_t_pos, best_t_neg = None, None
    num_guesses = None
    p = 0.05
    # # fixme: optimize for non_member scores first
    # cur_best_true_negatives = 0
    # for t_neg in tqdm(range(0, len(sorted_losses))):
    #     true_negatives = np.sum(sorted_labels[:(t_neg+1)] == 0)
    #     if true_negatives > cur_best_true_negatives

    for t_pos in tqdm(range(0, len(sorted_losses))):
        for t_neg in reversed(range(0, len(sorted_losses))):
            if t_neg < t_pos:
                continue
            true_positives = np.sum(sorted_labels[:(t_pos+1)] == 1)
            true_negatives = np.sum(sorted_labels[-t_neg:] == 0)
            cur_correct_predictions = true_positives + true_negatives
            if cur_correct_predictions > correct_predictions:
                correct_predictions = cur_correct_predictions
                best_t_pos = t_pos
                best_t_neg = t_neg
                num_guesses = (t_pos+1) + t_neg

    if num_guesses is not None:
        eps = get_eps_audit(len(all_labels), num_guesses, correct_predictions, p, delta)
    else:
        eps = None

    metric = {
        'audit_eps': eps, 'threshold_t_neg': sorted_losses[-best_t_neg],
        'threshold_t_pos': sorted_losses[(best_t_pos+1)],
        'best_accuracy': correct_predictions / num_guesses,
        'total_predictions': num_guesses, 'correct_predictions': correct_predictions
    }

    return metric


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
    threshold_range = np.arange(np.min(all_losses), np.max(all_losses) + 0.01, 0.05)
    # use fixed length threshold
    # threshold_range = np.linspace(start=np.min(all_losses), stop=np.max(all_losses) + 0.01, num=20)
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
