from sklearn.metrics import confusion_matrix


def get_confusion_matrix(all_labels, all_predictions):
    return confusion_matrix(all_labels, all_predictions).ravel()


def get_prec_rec_f1(tp, fp, fn):
    tpfp = tp + fp
    tpfn = tp + fn

    precision, recall, f1 = (0.0, 0.0, 0.0)
    if tpfp != 0:
        precision = tp / tpfp
    if tpfn != 0:
        recall = tp / tpfn
    if precision != 0 or recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1
