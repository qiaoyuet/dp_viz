import numpy as np
import math
import scipy


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
    threshold_range = np.arange(np.min(all_losses), np.max(all_losses) + 0.01, 0.01)
    results, recall = [], []
    best_accuracy = 0
    best_t_neg = 0
    total_predictions = 0
    correct_predictions = 0
    best_eps = 0
    p = 0.05
    for t_pos in threshold_range:
        positive_predictions = all_losses[all_losses <= t_pos]
        if len(positive_predictions) == 0:
            continue

        true_positives = np.sum(all_labels[all_losses <= t_pos] == 1)

        eps = get_eps_audit(len(all_labels), len(positive_predictions), true_positives, p, delta)
        precision = true_positives / len(positive_predictions)
        if eps > best_eps:
            print("EPSILON UPDATE:", eps)
            best_eps = eps
            best_t_pos = t_pos
        recalls = true_positives / np.sum(all_labels == 1)
        recall.append(recalls)

        # Step 2: With t_pos fixed, find t_neg that maximizes overall accuracy
        for t_neg in reversed(threshold_range):
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

            results.append({
                't_pos': t_pos,
                't_neg': best_t_neg,
                'best_precision': precision,
                'best_accuracy': best_accuracy,
                'recall': recall,
                'total_predictions': r,
                'correct_predictions': v
            })

    metric = {
        'best_eps': best_eps, 'threshold_t_neg': best_t_neg, 'threshold_t_pos': best_t_pos,
        'best_precision': best_precision, 'best_accuracy': best_accuracy
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

    return total_predictions, correct_predictions, len(all_losses), metric
