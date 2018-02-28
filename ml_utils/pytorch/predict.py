import logging
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm


def predict(test_loader, model):
    """Predict values for test_loader datas with model."""
    model.eval()
    predictions = []

    logging.info("Starting Prediction")
    for batch_idx, (data, _) in enumerate(tqdm(test_loader)):
        if torch.cuda.is_available():
            data = data.cuda(async=True)

        # Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.
        data = Variable(data, volatile=True)
        predictions.append(pred.data.cpu().numpy())
    return np.vstack(predictions)


def validate(valid_loader, model, metric):
    """Predict & compute a score with metric."""
    model.eval()
    predictions = []
    true_labels_binarized = []
    targets = []

    logging.info("Starting Validation")
    for batch_idx, (data, target) in enumerate(tqdm(valid_loader)):
        if torch.cuda.is_available():
            data, target = data.cuda(async=True), target.cuda(async=True)

        # Volatile variables do not save intermediate results and build graphs for backprop, achieving massive memory savings.
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)

        pred = model(data)
        predictions.append(pred.data.cpu().numpy())
        targets.append(target.data.cpu().numpy())

    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    score = metric(targets, predictions)
    logging.info("Score obtained: {}".format(score))
    return score
