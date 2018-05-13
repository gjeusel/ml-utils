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
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.

    Args:
        optimizer (torch.optim.something): your optimizer
        epoch (int): epoch number
        init_lr (float): initial learning_rate value
        lr_decay_epoch (int): learning rate modulo for which to decay learning rate.
        tb_writer (tensorboardX.SummaryWriter): for reporting purpose.

    Returns:
        optimizer with learning_rate updated
    """
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        logger.info('LR is set to {} for {}'.format(
            lr, optimizer.__class__.__name__))

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
        epoch (int): epoch number, used for learning_rate scheduling & log.
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

    loss_per_batch = []
    for batch_id, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(async=True), target.cuda(async=True)
        data = Variable(data)
        target = Variable(target, requires_grad=False)

        optimizer.zero_grad()
        output = net(data)

        loss = loss_func(output, target)
        if tb_writer:
            loss_per_batch.append(loss.cpu().item())
            tb_writer.add_scalar('data/loss_per_batch', loss.cpu().item(),
                                 batch_id * (epoch + 1))

        loss.backward()
        optimizer.step()
        if batch_id % 100 == 0:
            logger.info('Train epoch: {:03d} [{:05d}/{:05d} ({:.0f}%)]\tloss: {:.6f}'
                        '\t|cpu usage: {:.0f}%| |ram usage: {:.0f}%|'
                        .format(
                            epoch,
                            batch_id * len(data),
                            len(train_loader) * len(data),
                            100. * batch_id / len(train_loader),
                            loss.item(),
                            psutil.cpu_percent(),
                            psutil.virtual_memory().percent,
                        ))

    mean_loss = np.mean(loss_per_batch)
    logger.info('Mean Loss on {:03d}th epoch: {:.6f}'.format(epoch, mean_loss))

    return np.mean(loss_per_batch)


def train_da_rnn(epoch, train_loader,
                 encoder_optimizer, decoder_optimizer,
                 encoder, decoder,
                 loss_func,
                 tb_writer=None):

    encoder_optimizer = lr_scheduler(
        encoder_optimizer, epoch, tb_writer=tb_writer)
    decoder_optimizer = lr_scheduler(
        decoder_optimizer, epoch, tb_writer=tb_writer)

    iter_losses = []

    for batch_id, (X_batch, y_history, y_target, _) in enumerate(train_loader):

        if torch.cuda.is_available():
            X_batch = X_batch.cuda(async=True)
            y_history = y_history.cuda(async=True)
            y_target = y_target.cuda(async=True)

        X_batch = Variable(X_batch)
        y_history = Variable(y_history)
        y_target = Variable(y_target, requires_grad=False)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_weighted, input_encoded = encoder(X_batch)
        y_pred = decoder(input_encoded, y_history)

        loss = loss_func(y_pred, y_target)

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        iter_losses.append(loss.cpu().data[0])
        if tb_writer:
            tb_writer.add_scalar('data/batch_loss_epoch_{}'.format(epoch),
                                 loss.cpu().data[0], batch_id)

        if batch_id % 100 == 0:
            logger.info('Train epoch: {:03d} [{:05d}/{:05d} ({:.0f}%)]\tloss: {:.6f}'
                        '\t|cpu usage: {:.0f}%| |ram usage: {:.0f}%|'
                        .format(
                            epoch,
                            batch_id * X_batch.shape[0],
                            len(train_loader) * X_batch.shape[0],
                            100. * batch_id / len(train_loader),
                            loss.data[0],
                            psutil.cpu_percent(),
                            psutil.virtual_memory().percent,
                        ))

    return np.array(iter_losses).mean()
