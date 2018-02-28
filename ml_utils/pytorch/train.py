import logging
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
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logging.info('Train Epoch: {:03d} [{:05d}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader), loss.data[0]))
