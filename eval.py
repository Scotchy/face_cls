import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate(model, validation_loader, gt_labels, progress_bar=None):
    """Returns the ROC AUC and mean accuracy on validation set

    Args:
        model (nn.Module): The model to evaluate
        validation_loader (DataLoader): Validation loader
        gt_labels (array): Validation labels
        progress_bar (tqdm, optional): A tqdm progress bar to update. Defaults to None.

    Returns:
        float: ROC AUC
        float: Mean accuracy over the two classes
    """
    pre_labels = []
    for iteration, (images, labels) in enumerate(validation_loader):
        out = model(images.cuda())
        out = out.cpu().detach().numpy()
        pre_labels += [out]
        if progress_bar is not None:
            progress_bar.set_description(f"Validation : ({iteration}/{len(validation_loader)}")
    pre_labels = np.concatenate(pre_labels, axis=0).T[0]

    precision, _ = mean_acc_with_thresh(pre_labels, gt_labels, 20)
    auc = roc_auc_score(gt_labels, pre_labels)
    return auc, precision

def mean_acc_with_thresh(pre_labels, gt_labels, n_thresh):
    """FInd the best threshold to decide whether a sample belongs to 0 or 1 class and returns the mean accuracy

    Args:
        pre_labels (array): Predicted labels
        gt_labels (array): Validation labels
        n_thresh (int): Number of thresholds to test

    Returns:
        float: Mean accuracy (accuracy0 + accuracy1) / 2
    """
    thresholds = np.linspace(0, 1, n_thresh)

    accuracies = []
    
    for thresh in thresholds:
        pre = (pre_labels > thresh) * 1
        accuracies += [mean_accuracy(pre, gt_labels)]
    return np.max(accuracies), thresholds[np.argmax(accuracies)]

def mean_accuracy(predicted_class, gt_labels):
    indices0 =  gt_labels == 0
    indices1 = gt_labels == 1
    accuracy0 = np.sum(predicted_class[indices0] == gt_labels[indices0]) / np.sum(gt_labels == 0)
    accuracy1 = np.sum(predicted_class[indices1] == gt_labels[indices1]) / np.sum(gt_labels == 1)
    return (accuracy0 + accuracy1) / 2