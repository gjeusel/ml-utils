from pathlib import Path
import psutil
import logging
from datetime import datetime
import torch
from torch.autograd import Variable

logger = logging.getLogger(__name__)


def lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        logger.info('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def train(epoch, train_loader, net, loss_func, optimizer):
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
    optimizer = lr_scheduler(optimizer, epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(async=True), target.cuda(async=True)
        data = Variable(data)
        target = Variable(target, requires_grad=False)

        optimizer.zero_grad()
        output = net(data)

        loss = loss_func(output, target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info('Train Epoch: {:03d} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'
                         '\t|CPU usage: {:.0f}%| |RAM usage: {:.0f}%|'
                         .format(
                             epoch,
                             batch_idx * len(data), len(train_loader) * len(data),
                             100. * batch_idx / len(train_loader),
                             loss.data[0],
                             psutil.cpu_percent(),
                             psutil.virtual_memory().percent,
                         ))


def save_snapshot(epoch, net, score, optimizer, pth_dir=Path.cwd()):
    pth_name = '{net_name}_{now}_score_{score}_epoch_{epoch}.pth'.format(
        net_name=str(net.__class__.__name__),
        now=datetime.now().strftime('%Y-%M-%d-%H-%m'),
        score=score,
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
