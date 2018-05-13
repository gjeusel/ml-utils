import logging
import numpy as np
import torch
from sklearn.metrics import accuracy_score, fbeta_score, classification_report
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

logger = logging.getLogger(__name__)


def predict_image_classification(test_loader, net, score_type='proba'):
    """Predict values for test_loader datas with net."""
    net.eval()
    proba_pred = []
    class_pred = []

    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))

    logger.info("Starting Prediction")
    for batch_idx, (data, _) in enumerate(tqdm(test_loader)):
        if torch.cuda.is_available():
            data = data.cuda(async=True)

        data = Variable(data)

        pred = net(data)

        ppred = F.softmax(pred, dim=1).data.cpu()
        proba_pred.append(ppred.numpy())

        _, cpred = torch.max(pred.data.cpu(), dim=1)
        class_pred.append(cpred.numpy())

    proba_pred = np.concatenate(proba_pred)
    class_pred = np.concatenate(class_pred)

    if score_type == 'proba':
        return proba_pred
    elif score_type == 'class':
        return class_pred
    else:
        raise ValueError

    return np.concatenate(class_pred)


def validate_image_classification(valid_loader, net, score_func, score_type='proba'):
    """Predict & compute a accuracy_score & fbeta_score."""
    net.eval()
    proba_pred = []
    class_pred = []
    true_labels_binarized = []
    targets = []

    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(
            net, device_ids=range(torch.cuda.device_count()))

    logger.info("Starting Validation")
    for batch_idx, (data, target) in enumerate(tqdm(valid_loader)):
        if torch.cuda.is_available():
            data, target = data.cuda(async=True), target.cuda(async=True)

        data = Variable(data)
        target = Variable(target)

        pred = net(data)

        ppred = F.softmax(pred, dim=1).data.cpu()
        proba_pred.append(ppred.numpy())

        _, cpred = torch.max(pred.data.cpu(), dim=1)
        class_pred.append(cpred.numpy())

        targets.append(target.data.cpu().numpy())

    proba_pred = np.concatenate(proba_pred)
    class_pred = np.concatenate(class_pred)
    targets = np.concatenate(targets)
    # acc = accuracy_score(y_true=targets, y_pred=class_pred)
    # fbeta1 = fbeta_score(y_true=targets, y_pred=class_pred, beta=1)

    if score_type == 'proba':
        score = score_func(y_true=targets, y_pred=proba_pred)
    elif score_type == 'class':
        score = score_func(y_true=targets, y_pred=class_pred)
        logger.info(classification_report(y_true=targets, y_pred=class_pred))
    else:
        raise ValueError

    return score


def predict_da_rnn(test_loader, encoder, decoder):
    """Predict & compute a accuracy_score & fbeta_score."""
    predicted = []
    index = []

    logger.info("Starting Validation")
    for batch_id, (X_batch, y_history, _, timeindex) in enumerate(tqdm(test_loader)):

        if torch.cuda.is_available():
            X_batch = X_batch.cuda(async=True)
            y_history = y_history.cuda(async=True)

        # Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.
        X_batch = Variable(X_batch, volatile=True)
        y_history = Variable(y_history, volatile=True)

        _, input_encoded = encoder(X_batch)
        y_pred = decoder(input_encoded, y_history)

        y_pred = y_pred.data.cpu().numpy()
        predicted.append(y_pred.reshape(len(y_pred)))

        timeindex = timeindex.numpy()
        index.append(timeindex.reshape(len(timeindex)))

    timeindex = np.concatenate(index)
    predicted = np.concatenate(predicted)
    return timeindex, predicted


def validate_da_rnn(valid_loader, encoder, decoder, score_func):
    """Predict & compute a accuracy_score & fbeta_score."""
    targets = []
    predicted = []

    logger.info("Starting Validation")
    for batch_id, (X_batch, y_history, y_target, _) in enumerate(tqdm(valid_loader)):

        if torch.cuda.is_available():
            X_batch = X_batch.cuda(async=True)
            y_history = y_history.cuda(async=True)
            y_target = y_target.cuda(async=True)

        # Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.
        X_batch = Variable(X_batch, volatile=True)
        y_history = Variable(y_history, volatile=True)
        y_target = Variable(y_target, volatile=True)

        _, input_encoded = encoder(X_batch)
        y_pred = decoder(input_encoded, y_history)

        y_pred = y_pred.data.cpu().numpy()
        predicted.append(y_pred.reshape(len(y_pred)))

        y_target = y_target.data.cpu().numpy()
        targets.append(y_target.reshape(len(y_target)))

    predicted = np.concatenate(predicted)
    targets = np.concatenate(targets)

    score = score_func(y_true=targets, y_pred=predicted)
    return score
