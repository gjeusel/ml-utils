import logging
import numpy as np
import torch
from sklearn.metrics import accuracy_score, fbeta_score, classification_report
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

logger = logging.getLogger(__name__)

def predict(test_loader, net):
    """Predict values for test_loader datas with net."""
    net.eval()
    class_pred = []

    logger.info("Starting Prediction")
    for batch_idx, (data, _) in enumerate(tqdm(test_loader)):
        if torch.cuda.is_available():
            data = data.cuda(async=True)

        # Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.
        data = Variable(data, volatile=True)

        pred = net(data)

        _, cpred = torch.max(pred.data.cpu(), dim=1)
        class_pred.append(cpred.numpy())

    return np.concatenate(class_pred)


def validate(valid_loader, net):
    """Predict & compute a accuracy_score & fbeta_score."""
    net.eval()
    # proba_pred = []
    class_pred = []
    true_labels_binarized = []
    targets = []

    logger.info("Starting Validation")
    for batch_idx, (data, target) in enumerate(tqdm(valid_loader)):
        if torch.cuda.is_available():
            data, target = data.cuda(async=True), target.cuda(async=True)

        # Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)

        pred = net(data)

        # ppred = F.softmax(pred, dim=1).data.cpu()
        # proba_pred.append(ppred.numpy())

        _, cpred = torch.max(pred.data.cpu(), dim=1)
        class_pred.append(cpred.numpy())

        targets.append(target.data.cpu().numpy())

    # proba_pred = np.concatenate(proba_pred)
    class_pred = np.concatenate(class_pred)
    targets = np.concatenate(targets)
    acc = accuracy_score(y_true=targets, y_pred=class_pred)
    fbeta2 = fbeta_score(y_true=targets, y_pred=class_pred, beta=2)
    logger.info(classification_report(y_true=targets, y_pred=class_pred))
    return acc
