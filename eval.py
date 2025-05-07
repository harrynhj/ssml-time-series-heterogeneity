import numpy as np
from sklearn import metrics

from utils import get_bin_custom, CustomBins


# Evaluation for decompensation， in-hospital mortality
def print_metrics_binary(y_true, predictions, verbose=1):
    """
    Compute various metrics for binary classification with safety checks

    Args:
        y_true: Array of ground truth values (0 or 1)
        predictions: Array of model predictions (between 0 and 1)
        verbose: Whether to print metrics
    """
    # Safety checks
    if len(y_true) == 0 or len(predictions) == 0:
        print("Error: Empty arrays provided to print_metrics_binary")
        return 0, 0

    y_true = np.array(y_true)
    predictions = np.array(predictions)

    # Debug information
    print(f"Shape of y_true: {y_true.shape}, range: {np.min(y_true)}-{np.max(y_true)}")
    print(f"Shape of predictions: {predictions.shape}, range: {np.min(predictions)}-{np.max(predictions)}")

    # Ensure predictions are in the correct format
    if len(predictions.shape) == 1:
        # If predictions are just probability scores, convert to the expected format
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    # Make binary predictions
    binary_preds = predictions.argmax(axis=-1)

    # Check if all predictions are the same class
    if len(np.unique(binary_preds)) < 2 or len(np.unique(y_true)) < 2:
        print("Warning: All predictions or all ground truth values are the same class")
        # Return default values
        if len(np.unique(binary_preds)) < 2 and len(np.unique(y_true)) < 2:
            # If both are all the same class AND match, return perfect score
            if np.all(binary_preds == y_true):
                return 1.0, 1.0
            # Otherwise return worst score
            else:
                return 0.0, 0.0

    cf = metrics.confusion_matrix(y_true, binary_preds)
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    # Calculate metrics with safety checks
    acc = (cf[0][0] + cf[1][1]) / np.sum(cf) if np.sum(cf) > 0 else 0
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0]) if (cf[0][0] + cf[1][0]) > 0 else 0
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1]) if (cf[1][1] + cf[0][1]) > 0 else 0
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1]) if (cf[0][0] + cf[0][1]) > 0 else 0
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0]) if (cf[1][1] + cf[1][0]) > 0 else 0

    # AUROC calculation
    try:
        auroc = metrics.roc_auc_score(y_true, predictions[:, 1])
    except:
        print("Warning: Could not calculate AUROC, possibly due to single class")
        auroc = 0.5  # Default for random classifier

    # AUPRC calculation
    try:
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
        auprc = metrics.auc(recalls, precisions)
        minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    except:
        print("Warning: Could not calculate AUPRC")
        auprc = 0.0
        minpse = 0.0

    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))

    return auroc, auprc


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100

# Evaluation for length of stay
def print_metrics_regression(y_true, predictions, verbose=1):
    predictions = np.array(predictions)
    #predictions = np.max(predictions, axis=-1)#.flatten()
    y_true = np.array(y_true)
    # print(np.shape(predictions))
    # print(np.shape(y_true))

    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    # print()
    # print(333,np.shape(predictions),np.shape(y_true))
    # print(444,np.shape(prediction_bins),np.shape(y_true_bins))

    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("Custom bins confusion matrix:")
        print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins,
                                      weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)

    if verbose:
        print("Mean absolute deviation (MAD) = {}".format(mad))
        print("Mean squared error (MSE) = {}".format(mse))
        print("Mean absolute percentage error (MAPE) = {}".format(mape))
        print("Cohen kappa score = {}".format(kappa))
    return kappa, mad
