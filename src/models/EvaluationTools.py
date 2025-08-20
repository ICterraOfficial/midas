import numpy as np
from scipy.stats import norm
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def __get_worse_case(truths):
    """
    Internal function to determine the worst-case label for a list of truth labels.

    Parameters
    ----------
    truths : lists
        List of truth labels, where each inner list represents the truth labels for a single instance.

    Returns
    -------
    list of floats
        The worst-case label among the provided labels. Possible values are [0, 1], [1, 0], [0, 0], or [1, 1].
    """

    if [0., 1.] in truths:
        truth_label = [0., 1.]
    elif not [0., 1.] in truths and [1., 0.] in truths:
        truth_label = [1., 0.]
    elif not [0., 1.] in truths and [1., 0.] not in truths and [1., 1.] not in truths:
        truth_label = [0., 0.]
    else:
        truth_label = [1., 1.]

    return truth_label


def __combine_cc_mlo_labels(truths):
    """
    Combine the ground-truth labels of breast by applying a logical OR operation across the input labels.

    This function takes a list of ground-truth labels (each representing the presence or absence of a characteristic)
    and combines them by applying a logical OR operation across each label, resulting in a final combined label.

    Parameters
    ----------
    truths : list of lists or ndarray
        A list (or ndarray) where each inner list represents a set of ground-truth labels (binary values).
        Each list corresponds to the truth labels for a specific instance.

    Returns
    -------
    list of float
        A list representing the combined ground-truth label, where each element is the result of a logical OR operation
        along the respective axis. The combined label is converted to a float and returned as a list.

    Example
    -------
    >>> truths = [[1, 0], [0, 1]]
    >>> __combine_cc_mlo_labels(truths)
    [1.0, 1.0]
    """

    truth_label = np.any(truths, axis=0).astype(float).tolist()
    return truth_label


def get_truths_avgs(truths, predictions):
    """
    Calculate the average predictions and purged_truths truths per breast ID. This method returns array.

    Parameters
    ----------
    truths : dict
        Dictionary containing ground truths per breast ID.
    predictions : dict
        Dictionary containing predictions per breast ID.

    Returns
    -------
    tuple
        A tuple containing two lists:
        - purged_truths : list
            List of purged truth labels per breast ID.
        - avg_preds : list
            List of average predictions per breast ID.
    """

    avg_preds = []
    purged_truths = []

    for key, preds in predictions.items():
        preds = np.array(preds)
        # Calculate the average prediction per breast
        avg_pred = np.average(preds, axis=0)
        # If malign is exist return [0, 1] or malign and benign exist together [1, 1]
        truth_label = __combine_cc_mlo_labels(truths[key])

        avg_preds.append(avg_pred)
        purged_truths.append(truth_label)

    return np.array(purged_truths), np.array(avg_preds)


def calculate_roc_pr_auc(truths, predictions):
    """
    Calculate the ROC and PR (Precision-Recall) AUC scores.

    Parameters
    ----------
    truths : np.ndarray
        Numpy array of ground truth labels.
    predictions : np.ndarray
        Numpy array of predicted probabilities.

    Returns
    -------
    dict
        A dictionary with two keys: 'roc' and 'pr'. Each contains metrics related to the respective curve.

        - 'roc' : dict
            Dictionary containing:
                - 'fprs' : np.ndarray
                    False Positive Rates computed at various thresholds.
                - 'tprs' : np.ndarray
                    True Positive Rates computed at various thresholds.
                - 'auc' : float
                    Area Under the ROC Curve (ROC AUC), representing the model's ability to distinguish between classes.

        - 'pr' : dict
            Dictionary containing:
                - 'precisions' : np.ndarray
                    Precision values computed at various thresholds.
                - 'recalls' : np.ndarray
                    Recall values computed at various thresholds.
                - 'auc' : float
                    Area Under the Precision-Recall Curve (PR AUC), useful especially for imbalanced datasets.
    """

    fprs, tprs, thresholds = roc_curve(y_true=truths, y_score=predictions)
    roc_auc = auc(fprs, tprs)
    precisions, recalls, thresholds = precision_recall_curve(y_true=truths, probas_pred=predictions)
    pr_auc = auc(recalls, precisions)

    metrics = {
        'roc': {'fprs': fprs,
                'tprs': tprs,
                'auc': roc_auc},
        'pr': {'precisions': precisions,
               'recalls': recalls,
               'auc': pr_auc}
    }

    return metrics


def calculate_auc_ci(auc, positive_gts, confidence=0.95):
    """See details: https://real-statistics.com/descriptive-statistics/roc-curve-classification-table/auc-confidence-interval/
    """

    # Get the total number of Positve and Negative samples
    n_pos = np.count_nonzero(positive_gts)
    n_neg = len(positive_gts) - n_pos

    z_score = norm.ppf(1 - (1 - confidence) / 2)

    # AUC Variance
    q0 = auc * (1 - auc)
    # AUC Variance w.r.t. number of negative samples.
    q1 = auc / (2 - auc) - auc ** 2
    # AUC Variance w.r.t. number of positive samples.
    q2 = (2 * (auc ** 2)) / (1 + auc) - auc ** 2

    # Calculate standard error.
    se = np.sqrt(((q0 + (n_neg) * q1 + (n_pos) * q2)) / (n_neg * n_pos))

    lower_bound = auc - z_score * se
    upper_bound = auc + z_score * se

    lower_bound = round(lower_bound, 2)
    upper_bound = round(upper_bound, 2)

    return lower_bound, upper_bound


def calculate_pr_baseline(positive_gts):
    """
    Calculate the baseline PR AUC score for a given set of positive ground truth labels.

    Parameters
    ----------
    positive_gts: np.ndarray

    Returns
    -------
    pr_base: float
    """

    # Calculate Precision-Recall baseline
    n_pos = np.count_nonzero(positive_gts)
    pr_base = n_pos / len(positive_gts)

    return pr_base


class MetricCalculator:
    """
    A class to handle the storage of model predictions and ground truths, and to calculate evaluation metrics.

    This class is designed to store predictions and ground truths for each breast sample, indexed by breast ID.
    It provides methods to store these values and calculate various evaluation metrics, such as ROC AUC and
    Precision-Recall AUC, based on the stored data.

    Attributes
    ----------
    y_truths : dict
        A dictionary where keys are breast IDs and values are lists of ground truth labels for each breast sample.
    y_preds : dict
        A dictionary where keys are breast IDs and values are lists of model predictions for each breast sample.

    Methods
    -------
    store_preds_truths(breast_ids, predictions, truths)
        Stores predictions and ground truths for each breast sample identified by breast IDs.
    calculate_metrics()
        Calculates and returns evaluation metrics based on the stored predictions and ground truths.
    clear_cache()
        Clears the stored predictions and ground truths.
    """

    def __init__(self):
        """
        Initialize the MetricCalculator class with empty dictionaries for storing predictions and ground truths.

        Attributes
        ----------
        y_truths : dict
            Initialized as an empty dictionary to store ground truth labels for each breast sample, indexed by breast ID.
        y_preds : dict
            Initialized as an empty dictionary to store model predictions for each breast sample, indexed by breast ID.
        """

        self.y_truths = {}
        self.y_preds = {}

    def store_preds_truths(self, breast_ids, predictions, truths):
        """
        Store the predictions and ground truths for each breast sample.

        Parameters
        ----------
        breast_ids : list of str
            Identifiers for the breast samples in the batch.
        predictions : torch.Tensor
            Model predictions for the current batch.
        truths : torch.Tensor
            Ground truth labels for the current batch.
        """

        predictions = predictions.squeeze(1)

        for i, breast_id in enumerate(breast_ids):
            if breast_id in self.y_preds.keys():
                self.y_preds[breast_id].append(predictions[i].cpu().tolist())
            else:
                self.y_preds[breast_id] = []
                self.y_preds[breast_id].append(predictions[i].cpu().tolist())

            if breast_id in self.y_truths.keys():
                self.y_truths[breast_id].append(truths[i].tolist())
            else:
                self.y_truths[breast_id] = []
                self.y_truths[breast_id].append(truths[i].tolist())

    def calculate_metrics(self, positive_class_dim=1, clear_cache=False):
        """
        Calculate evaluation metrics based on the stored predictions and ground truths.

        Parameters
        ----------
        positive_class_dim: int
            Which dimension is considered a positive class.
        clear_cache : bool, optional
            If True, clears the stored predictions and ground truths after calculating the metrics. Default is False.

        Returns
        -------
        dict
            A dictionary with two keys: 'roc' and 'pr'. Each contains metrics related to the respective curve.

            - 'roc' : dict
                Dictionary containing:
                    - 'fprs' : np.ndarray
                        False Positive Rates computed at various thresholds.
                    - 'tprs' : np.ndarray
                        True Positive Rates computed at various thresholds.
                    - 'auc' : float
                        Area Under the ROC Curve (ROC AUC), representing the model's ability to distinguish between
                        classes.
                    - 'lower': float
                        Lower bound of the ROC AUC.
                    - 'upper': float
                        Upper bound of the ROC AUC.

            - 'pr' : dict
                Dictionary containing:
                    - 'precisions' : np.ndarray
                        Precision values computed at various thresholds.
                    - 'recalls' : np.ndarray
                        Recall values computed at various thresholds.
                    - 'auc' : float
                        Area Under the Precision-Recall Curve (PR AUC), useful especially for imbalanced datasets.
                    - 'lower': float
                        Lower bound of the PR AUC.
                    - 'upper': float
                        Upper bound of the PR AUC.
                    - 'baseline': float
                        The ratio of the number of positives to the total number of samples.

            Additional metrics may be included in future versions.
        """

        assert self.y_truths != {} and self.y_preds != {}, ('y_truths and y_preds are empty. '
                                                            'Re-run store_preds_truths and try clear_cache==False.')

        purged_truths, avg_preds = get_truths_avgs(self.y_truths, self.y_preds)

        # Define positive class and get corresponding values for the class.
        positive_truths, positive_preds = purged_truths[:, positive_class_dim], avg_preds[:, positive_class_dim]

        # Calculate metrics.
        metrics = calculate_roc_pr_auc(positive_truths, positive_preds)

        # Calculate confidence intervals.
        roc_low, roc_up = calculate_auc_ci(metrics['roc']['auc'], positive_truths)
        pr_low, pr_up = calculate_auc_ci(metrics['pr']['auc'], positive_truths)
        pr_base = calculate_pr_baseline(positive_truths)

        metrics['roc'].update({'lower': roc_low, 'upper': roc_up})
        metrics['pr'].update({'lower': pr_low, 'upper': pr_up})
        metrics['pr'].update({'baseline': pr_base})

        if clear_cache:
            self.clear_cache()

        return metrics

    def clear_cache(self):
        """
        Clears the stored predictions and ground truths.
        """

        self.y_truths = {}
        self.y_preds = {}
