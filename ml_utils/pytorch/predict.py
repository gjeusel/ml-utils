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

        # Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.
        data = Variable(data, volatile=True)

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

        # Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)

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
    # fbeta2 = fbeta_score(y_true=targets, y_pred=class_pred, beta=2)

    if score_type == 'proba':
        score = score_func(y_true=targets, y_pred=proba_pred)
    elif score_type == 'class':
        score = score_func(y_true=targets, y_pred=class_pred)
    else:
        raise ValueError

    logger.info(classification_report(y_true=targets, y_pred=class_pred))
    return score


def predict_timeserie(X, encoder, decoder,
                      train_size, n_timestep, batch_size):
    y_pred = np.zeros(X.shape[0] - train_size)

    logger.info("Starting Prediction")

    range_main_loop = range(0, len(y_pred), batch_size)  # batch_size step
    for i, _ in enumerate(tqdm(range_main_loop)):
        batch_idx = np.array(range(len(y_pred)))[i : (i + batch_size)]
        X_batch = np.zeros((len(batch_idx), n_timestep - 1, X.shape[1]))
        y_history = np.zeros((len(batch_idx), n_timestep - 1))

        # Construct X_batch
        for j in range(len(batch_idx)):
            local_range = range(
                batch_idx[j] + train_size - n_timestep,
                batch_idx[j] + train_size - 1)

            X_batch[j, :, :] = X[local_range, :]
            y_history[j, :] = y[local_range]

        X_batch = Variable(torch.from_numpy(X_batch).type(torch.FloatTensor))
        y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor))

        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
            y_history = y_history.cuda()

        _, input_encoded = encoder(X_batch)
        y_pred[i:(i + batch_size)] = decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]

    return y_pred


def validate_timeserie(X, y, encoder, decoder,
                       train_size, n_timestep, batch_size):
    y_pred = np.zeros(train_size - n_timestep + 1)

    logger.info("Starting Validation")

    range_main_loop = range(0, len(y_pred), batch_size)  # batch_size step
    for i, _ in enumerate(tqdm(range_main_loop)):
        batch_idx = np.array(range(len(y_pred)))[i : (i + batch_size)]
        X_batch = np.zeros((len(batch_idx), n_timestep - 1, X.shape[1]))
        y_history = np.zeros((len(batch_idx), n_timestep - 1))

        # Construct X_batch
        for j in range(len(batch_idx)):
            local_range = range(batch_idx[j], batch_idx[j] + n_timestep - 1)
            X_batch[j, :, :] = X[local_range, :]
            y_history[j, :] = y[local_range]

        X_batch = Variable(torch.from_numpy(X_batch).type(torch.FloatTensor))
        y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor))

        if torch.cuda.is_available():
            X_batch = X_batch.cuda()
            y_history = y_history.cuda()

        _, input_encoded = encoder(X_batch)
        y_pred[i:(i + batch_size)] = decoder(input_encoded, y_history).cpu().data.numpy()[:, 0]

    return y_pred
