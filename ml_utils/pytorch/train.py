from pathlib import Path
import psutil
import logging
from datetime import datetime
import torch
from torch.autograd import Variable


def lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        logging.info('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def train(epoch, train_loader, model, loss_func, optimizer):
    """Unique epoch computation.

    Args:
        epoch (int): epoch number, used for learning_rate scheduling.
        train_loader (torch.utils.data.dataloader.DataLoader): data loader for
            training purpose.
        model: your neural network
        loss_func (torch.nn.modules.loss.something): your loss function
        optimizer (torch.optim.something): your optimizer

    .. _Pytorch Net Example:
        http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    """
    model.train()  # set a bool to true, this has ony effect on dropout, batchnorm etc..
    optimizer = lr_scheduler(optimizer, epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(async=True), target.cuda(async=True)
        data = Variable(data)
        target = Variable(target, requires_grad=False)

        optimizer.zero_grad()
        output = model(data)

        loss = loss_func(output, target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logging.info('Train Epoch: {:03d} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'
                         '\t|CPU usage: {:.0f}%| |RAM usage: {:.0f}%|'
                         .format(
                             epoch,
                             batch_idx * len(data), len(train_loader) * len(data),
                             100. * batch_idx / len(train_loader),
                             loss.data[0],
                             psutil.cpu_percent(),
                             psutil.virtual_memory().percent,
                         ))


def save_snapshot(epoch, net, loss, optimizer, fout_name=None):
    if fout_name is None:
        fout_name = '{model_name}_{now}_loss_{loss}_epoch_{epoch}.pth'.format(
            model_name=str(model.__class__.__name__),
            now=datetime.now().strftime('%Y-%M-%d'),
            loss=loss,
            epoch=epoch,
        )
        fout_name = Path.cwd() / fout_name

        state = {
            'epoch': epoch,
            'net': net.state_dict(),
            'loss': loss,
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, fout_name.as_posix())
        logging.info("Snapshot saved to {}".format(fout_name.as_posix()))


def load_snapshot(fname):
    state_dict = torch.load(fname.as_posix)
    epoch = state_dict['epoch']
    net_state_dict = state_dict['net']
    loss = state_dict['loss']
    optimizer_state_dict = state_dict['optimizer']

    return epoch, net_state_dict, loss, optimizer_state_dict
