from pathlib import Path
import psutil
import logging
from datetime import datetime
import torch
from torch.autograd import Variable

import numpy as np

logger = logging.getLogger(__name__)


def lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=7,
                 tb_writer=None):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        logger.info('LR is set to {} for {}'.format(lr, optimizer.__class__.__name__))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if tb_writer:
        tb_writer.add_scalar('data/learning_rate', lr, epoch)

    return optimizer


def save_snapshot(epoch, net, score, optimizer, pth_dir=Path.cwd()):
    pth_name = '{net_name}_{now}_score_{score}_epoch_{epoch}.pth'.format(
        net_name=str(net.__class__.__name__),
        now=datetime.now().strftime('%Y-%M-%d-%H-%m'),
        score=round(score, 3),
        epoch=epoch,
    )
    pth_path = pth_dir / pth_name

    logger.info("Saving snapshot to {}...".format(pth_path.as_posix()))

    state = {
        'epoch': epoch,
        'net': net.state_dict(),
        'score': score,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, pth_path.as_posix())


def load_snapshot(pth_path):
    """Load a Snapshot given a path to a .pth file."""
    if hasattr(pth_path, 'as_posix'):
        state_dict = torch.load(pth_path.as_posix())
        logger.info("Loading snapshot {}...".format(pth_path.as_posix()))
    elif isinstance(pth_path, str):
        try:
            state_dict = torch.load(pth_path)
        except:
            raise ValueError
        logger.info("Loading snapshot {}...".format(pth_path))


    epoch = state_dict['epoch']
    net_state_dict = state_dict['net']
    score = state_dict['score']
    optimizer_state_dict = state_dict['optimizer']

    return epoch, net_state_dict, score, optimizer_state_dict


def train_image_classification(epoch, train_loader, net, loss_func, optimizer,
                               tb_writer=None):
    """Unique epoch computation.

    Args:
        epoch (int): epoch number, used for learning_rate scheduling.
        train_loader (torch.utils.data.dataloader.DataLoader): data loader for
            training purpose.
        net: your neural network
        loss_func (torch.nn.modules.loss.something): your loss function
        optimizer (torch.optim.something): your optimizer

    .. _Pytorch Net Example:
        http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    """
    net.train()  # set a bool to true, this has ony effect on dropout, batchnorm etc..
    optimizer = lr_scheduler(optimizer, epoch, tb_writer=tb_writer)

    for batch_index, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(async=True), target.cuda(async=True)
        data = Variable(data)
        target = Variable(target, requires_grad=False)

        optimizer.zero_grad()
        output = net(data)

        loss = loss_func(output, target.long())
        if tb_writer:
            tb_writer.add_scalar('data/loss', loss.cpu().data[0],
                                  batch_index * (epoch + 1))

        loss.backward()
        optimizer.step()
        if batch_index % 100 == 0:
            logger.info('Train epoch: {:03d} [{:04d}/{:04d} ({:.0f}%)]\tloss: {:.6f}'
                         '\t|cpu usage: {:.0f}%| |ram usage: {:.0f}%|'
                         .format(
                             epoch,
                             batch_index * len(data),
                             len(train_loader) * len(data),
                             100. * batch_index / len(train_loader),
                             loss.data[0],
                             psutil.cpu_percent(),
                             psutil.virtual_memory().percent,
                         ))


def train_oneiter_da_rnn(X, y_history, y_target,
                         encoder_optimizer, decoder_optimizer,
                         encoder, decoder,
                         loss_func, tb_writer=None):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    X = Variable(torch.from_numpy(X).type(torch.FloatTensor))
    y_history = Variable(torch.from_numpy(y_history).type(torch.FloatTensor))
    y_true = Variable(torch.from_numpy(y_target).type(torch.FloatTensor))

    if torch.cuda.is_available():
        X = X.cuda()
        y_history = y_history.cuda()
        y_true = y_true.cuda()

    input_weighted, input_encoded = encoder(X)
    y_pred = decoder(input_encoded, y_history)

    loss = loss_func(y_pred, y_true)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]


def train_oneepoch_da_rnn(epoch, X, y, train_size,
                          batch_size, n_timestep, perm_idx,
                          encoder_optimizer, decoder_optimizer,
                          encoder, decoder,
                          loss_func, tb_writer=None):

    encoder_optimizer = lr_scheduler(encoder_optimizer, epoch, tb_writer=tb_writer)
    decoder_optimizer = lr_scheduler(decoder_optimizer, epoch, tb_writer=tb_writer)

    batches_range = range(0, train_size, batch_size)[:-1]  # exclude last element
    iter_losses = []

    for batch_id, j in enumerate(batches_range):
        batch_index = perm_idx[j: (j + batch_size)]

        X_batch = np.zeros((len(batch_index), n_timestep - 1, X.shape[1]))
        y_history = np.zeros((len(batch_index), n_timestep - 1))
        y_target = y[batch_index + n_timestep]

        for k in range(len(batch_index)):
            X_batch[k, :, :] = X[batch_index[k] : (batch_index[k] + n_timestep - 1), :]
            y_history[k, :] = y[batch_index[k] : (batch_index[k] + n_timestep - 1)]

        loss = train_oneiter_da_rnn(X_batch, y_history, y_target,
                                    encoder_optimizer, decoder_optimizer,
                                    encoder, decoder, loss_func, tb_writer)
        iter_losses.append(loss)
        if batch_id % 100 == 0:
            logger.info('Train epoch: {:03d} [{:04d}/{:04d} ({:.0f}%)]\tloss: {:.6f}'
                            '\t|cpu usage: {:.0f}%| |ram usage: {:.0f}%|'
                            .format(
                                epoch, batch_id, len(batches_range),
                                100 * batch_id / len(batches_range),
                                loss,
                                psutil.cpu_percent(),
                                psutil.virtual_memory().percent,
                            ))
    return np.array(iter_losses).mean()
