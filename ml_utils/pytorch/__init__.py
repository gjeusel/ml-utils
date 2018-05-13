import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
# https://github.com/pytorch/pytorch/issues/973

import logging

import pandas as pd

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.metrics import log_loss, mean_squared_error

from .profiling import Timer

from .train import save_snapshot, load_snapshot

# Image Classification purpose:
from .dataset import CustomDataset, train_valid_split
from .train import train_image_classification
from .predict import predict_image_classification, validate_image_classification

# Timeserie Forecast
from .dataset import TsFcstDataset
from .dual_stagged_attention import encoder, decoder
from .train import train_da_rnn
from .predict import predict_da_rnn, validate_da_rnn

logger = logging.getLogger(__file__)


class ImageClassification():
    """Wrapper class for image classification."""

    def __init__(self, train_csvpath, train_imgdir, snapshot_dir,
                 test_csvpath=None, test_imgdir=None, sub_dir=None,
                 batch_size=4, num_workers=4,
                 tb_writer=None,
                 ):
        """
        Args:
            train_csvpath (pathlib.Path): path to csv with train informations.
            train_imgdir (pathlib.Path): path to images for train purpose.
            test_csvpath (pathlib.Path): path to csv with test informations.
                Defaults to None if you won't predict for unknown labels.
            test_imgdir (pathlib.Path): path to images for test purpose.
                Defaults to None if you won't predict for unknown labels.
            sub_dir (pathlib.Path): path for submission file outputs.
            batch_size (int): size of batch.
            num_workers (int): number of workers.
            tb_writer (tensorboardX.SummaryWriter): for reporting purpose.
                Defaults to None if you don't care.
        """

        self.train_csvpath = train_csvpath
        self.train_imgdir = train_imgdir
        self.snapshot_dir = snapshot_dir

        self.test_csvpath = test_csvpath
        self.test_imgdir = test_imgdir
        self.sub_dir = sub_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tb_writer = tb_writer

    def set_train_loaders(self,
                          ds_transform_augmented, ds_transform_raw,
                          sampler=SubsetRandomSampler,
                          perc_train_valid=0.1,
                          multilabel=False,
                          debug=False,
                          limit_load_at_debug=100,
                          ):
        """Set all variables to handle images loaders for train purpose.

        Args:
            ds_transform_augmented (torchvision.transforms.Compose): image transformations
                to applies for training images. (ds is for dataset)
            ds_transform_raw (torchvision.transforms.Compose): image transformations
                to applies for validation and/or test sets. (With no image augmentation
                techniques like random flips or rotation ...)
            perc_train_valid (float): percentage of datas to keep for validation purpose.
            multilabel (bool): if is a multilabel classification problem.
            debug (bool): if debug, then limit number of images loaded to limit_load_at_debug.
        """

        self.ds_transform_augmented = ds_transform_augmented
        self.ds_transform_raw = ds_transform_raw

        # Loading the dataset
        limit_load = limit_load_at_debug if debug else None
        X_train = CustomDataset(self.train_csvpath, self.train_imgdir,
                                transform=ds_transform_augmented,
                                limit_load=limit_load,
                                multilabel=multilabel,
                                )
        X_val = CustomDataset(self.train_csvpath, self.train_imgdir,
                              transform=ds_transform_raw,
                              limit_load=limit_load,
                              multilabel=multilabel,
                              )

        # Creating a validation split
        train_idx, valid_idx = train_valid_split(X_train, perc_train_valid)

        if sampler is not None:
            train_sampler = sampler(train_idx)
            valid_sampler = sampler(valid_idx)
        else:
            train_sampler, valid_sampler = None, None

        # Both dataloader loads from the same dataset but with different indices
        train_loader = DataLoader(X_train,
                                  sampler=train_sampler,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  # pin_memory=True,
                                  )

        valid_loader = DataLoader(X_val,
                                  sampler=valid_sampler,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  # pin_memory=True,
                                  )

        self.X_train, self.X_val = X_train, X_val
        self.train_idx, self.valid_idx = train_idx, valid_idx
        self.train_loader, self.valid_loader = train_loader, valid_loader

    def set_test_loaders(self, ds_transform_raw, multilabel=False,
                         debug=False, limit_load_at_debug=100):

        if None in [self.test_csvpath, self.test_imgdir]:
            raise ValueError('You forgot to set either test_csvpath or test_imgdir')

        limit_load = limit_load_at_debug if debug else None
        X_test = CustomDataset(self.test_csvpath, self.test_imgdir,
                               transform=ds_transform_raw,
                               limit_load=limit_load)

        test_loader = DataLoader(X_test,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 # pin_memory=True,
                                 )

        self.X_test, self.test_loader = X_test, test_loader


    def train(self, epochs, net, loss_func, optimizer,
              score_func, score_higher_is_better=True, score_type='proba'):
        """Train the network.

        Args:
            epochs (int): number of epochs max for whichto iterate.
            net (Upper Class of torch.nn.Module): your networks.
            loss_func (function): your loss function (among torch.nn loss functions for example.)
            optimizer (torch.optim.something): your optimizer
            score_func (function): used to compute a score at validation time.
            score_higher_is_better (bool): should the score be considerated better
                if higher or not.
            score_type (str): among ['proba', 'class'], used at validation time.
                Describe how the output of the network should be considerated depending
                on the score_func choosen.
                Defaults to 'proba'.
        """

        if torch.cuda.is_available():
            net.cuda()
            net = torch.nn.DataParallel(
                net, device_ids=range(torch.cuda.device_count()))

        best_score = 1.0
        for epoch in range(epochs):
            with Timer('Epoch {}'.format(epoch)):

                # Train and validate
                train_image_classification(
                    epoch, self.train_loader, net, loss_func, optimizer,
                    self.tb_writer)

                score = validate_image_classification(self.valid_loader, net,
                                                      score_func=score_func,
                                                      score_type=score_type,
                                                      )

                if self.tb_writer:
                    self.tb_writer.add_scalar('data/score', score, epoch)

                if score_higher_is_better:
                    is_better = best_score < score
                else:
                    is_better = best_score > score

                if is_better and epoch > 10:
                    best_score = score
                    save_snapshot(epoch+1, net, score,
                                  optimizer, self.snapshot_dir)

        self.net = net

    def continue_training(self, pth_path, epochs, net, loss_func, optimizer,
                          score_func, score_higher_is_better=True, score_type='proba'):
        """Continue training a network.

        Args:
            pth_path (pathlib.Path): path to snapshot.
            epochs (int): number of epochs max for whichto iterate.
            net (Upper Class of torch.nn.Module): same networks used to obtain pth_path snapshot.
            loss_func (function): your loss function (among torch.nn loss functions for example.)
            optimizer (torch.optim.something): your optimizer
            score_func (function): used to compute a score at validation time.
            score_higher_is_better (bool): should the score be considerated better
                if higher or not.
            score_type (str): among ['proba', 'class'], used at validation time.
                Describe how the output of the network should be considerated depending
                on the score_func choosen.
                Defaults to 'proba'.
        """
        epoch_start, net_state_dict, score, optimizer_state_dict = load_snapshot(
            pth_path)
        assert epochs >= epoch_start

        net.load_state_dict(net_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

        if torch.cuda.is_available():
            net.cuda()
            net = torch.nn.DataParallel(
                net, device_ids=range(torch.cuda.device_count()))

        for epoch in range(epoch_start, epochs):
            with Timer('Epoch {}'.format(epoch)):

                # Train and validate
                train_image_classification(
                    epoch, self.train_loader, net, loss_func, optimizer)

                score = validate_image_classification(self.valid_loader, net)

                if self.tb_writer:
                    self.tb_writer.add_scalar('data/score', score, epoch)

                if score_higher_is_better:
                    is_better = best_score < score
                else:
                    is_better = best_score > score

                if is_better and epoch > 10:
                    best_score = score
                    save_snapshot(epoch+1, net, score,
                                  optimizer, self.snapshot_dir)

    def predict_for_submission(self, net, pth_path,
                               ds_transform_raw, multilabel=False,
                               output_type='proba',
                               batch_size=4, num_workers=4, debug=False):
        """Predict """

        self.set_test_loaders(ds_transform_raw=ds_transform_raw, multilabel=multilabel)

        # Load net
        epoch, net_state_dict, score, _ = load_snapshot(pth_path)
        net.load_state_dict(net_state_dict)

        # Predict
        y_pred = predict_image_classification(test_loader, net, output_type=output_type)

        return y_pred

    def write_submission_file(self, predicted, labels):
        # Submission
        X_test = CustomDataset(self.test_csvpath, self.test_imgdir)
        df_sub = X_test.df
        df_sub[labels] = predicted

        sub_name = 'submission_{net_name}_{now}_score_{score}_epoch_{epoch}.csv'.format(
            net_name=str(net.__class__.__name__),
            now=datetime.now().strftime('%Y-%M-%d-%H-%m'),
            score=score,
            epoch=epoch,
        )
        sub_path = sub_dir / sub_name
        df_sub.to_csv(sub_path)
        logger.info('Submission file saved in {}'.format(sub_path))


class TimeserieFcst_DA_RNN():
    """Wrapper class for timeserie forecasting using Dual-Stage Attention-Based RNN."""

    def __init__(self, n_timestep=10, snapshot_dir=None, sub_dir=None,
                 batch_size=4, num_workers=4,
                 tb_writer=None):

        self.n_timestep = 10

        self.snapshot_dir = snapshot_dir
        self.sub_dir = sub_dir

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tb_writer = tb_writer

    def set_train_loaders(self, df, target_col,
                          sampler=SubsetRandomSampler,
                          perc_train_valid=0.1):

        self.df = df

        dataset = TsFcstDataset(n_timestep=self.n_timestep,
                                df=df, target_col=target_col)

        # Creating a validation split
        train_idx, valid_idx = train_valid_split(dataset, perc_train_valid,
                                                 shuffle=False)

        if sampler is not None:
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
        else:
            train_sampler, valid_sampler = None, None

        # Both dataloader loads from the same dataset but with different indices
        train_loader = DataLoader(dataset,
                                  sampler=train_sampler,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  # pin_memory=True,
                                  )

        valid_loader = DataLoader(dataset,
                                  sampler=valid_sampler,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  # pin_memory=True,
                                  )

        self.train_idx, self.valid_idx = train_idx, valid_idx
        self.train_loader, self.valid_loader = train_loader, valid_loader

        self.ncols = dataset.df.shape[1]  # used at encoder init time

        df_train = df.iloc[train_idx]
        df_valid = df.iloc[valid_idx]
        return df_train, df_valid

    def set_encoder_decoder(self,
                            encoder_hidden_size=64, decoder_hidden_size=64,
                            learning_rate=0.01,
                            ):

        self.encoder = encoder(input_size=self.ncols-1,
                               hidden_size=encoder_hidden_size,
                               n_timestep=self.n_timestep)

        self.decoder = decoder(encoder_hidden_size=encoder_hidden_size,
                               decoder_hidden_size=decoder_hidden_size,
                               n_timestep=self.n_timestep)

        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad,
                          self.encoder.parameters()),
            lr=learning_rate)
        self.decoder_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad,
                          self.decoder.parameters()),
            lr=learning_rate)

    def train(self, epochs=10, loss_func=nn.MSELoss()):

        best_score = 10**10
        for epoch in range(epochs):
            with Timer('Epoch {}'.format(epoch)):
                epoch_loss = train_da_rnn(epoch, self.train_loader,
                                          self.encoder_optimizer, self.decoder_optimizer,
                                          self.encoder, self.decoder, loss_func,
                                          self.tb_writer)

                score = validate_da_rnn(self.valid_loader,
                                        self.encoder, self.decoder,
                                        score_func=mean_squared_error)

            logger.info('Mean epoch {} loss: {:.6f}'.format(epoch, epoch_loss))
            logger.info('Score epoch {}: {:.6f}'.format(epoch, score))

            if self.tb_writer:
                self.tb_writer.add_scalar('data/epoch_loss', epoch_loss, epoch)
                self.tb_writer.add_scalar('data/score', score, epoch)

            if best_score > score and epoch > 4:
                best_score = score
                save_snapshot(epoch+1, self.encoder, score, self.encoder_optimizer,
                              self.snapshot_dir)
                save_snapshot(epoch+1, self.decoder, score, self.decoder_optimizer,
                              self.snapshot_dir)

    def continue_training(self, epochs,
                          encoder_pth_path, decoder_pth_path,
                          loss_func=nn.MSELoss()):
        """Continue training."""
        epoch_start, encoder_state_dict, score, encoder_optimizer_state_dict = load_snapshot(
            encoder_pth_path)
        epoch_start, decoder_state_dict, score, decoder_optimizer_state_dict = load_snapshot(
            decoder_pth_path)

        assert epochs >= epoch_start

        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)

        self.encoder_optimizer.load_state_dict(encoder_optimizer_state_dict)
        self.decoder_optimizer.load_state_dict(decoder_optimizer_state_dict)

        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        best_score = 10**10
        for epoch in range(epoch_start, epochs):
            with Timer('Epoch {}'.format(epoch)):
                epoch_loss = train_da_rnn(epoch, self.train_loader,
                                          self.encoder_optimizer, self.decoder_optimizer,
                                          self.encoder, self.decoder, loss_func,
                                          self.tb_writer)

                score = validate_da_rnn(self.valid_loader,
                                        self.encoder, self.decoder,
                                        score_func=mean_squared_error)

            logger.info('Mean epoch {} loss: {:.6f}'.format(epoch, epoch_loss))
            logger.info('Score epoch {}: {:.6f}'.format(epoch, score))

            if self.tb_writer:
                self.tb_writer.add_scalar('data/epoch_loss', epoch_loss, epoch)
                self.tb_writer.add_scalar('data/score', score, epoch)

            if best_score > score and epoch > 4:
                best_score = score
                save_snapshot(epoch+1, self.encoder, score, self.encoder_optimizer,
                              self.snapshot_dir)
                save_snapshot(epoch+1, self.decoder, score, self.decoder_optimizer,
                              self.snapshot_dir)

    def predict(self, encoder_pth_path, decoder_pth_path,
                df, target_col):

        dataset = TsFcstDataset(n_timestep=self.n_timestep, target_col=target_col,
                                df=df, test_mode=True)

        test_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 # pin_memory=True,
                                 )

        self.test_loader = test_loader

        epoch, encoder_state_dict, score, _ = load_snapshot(
            encoder_pth_path)
        epoch, decoder_state_dict, score, _ = load_snapshot(
            decoder_pth_path)

        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)

        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        timeindex, predicted = predict_da_rnn(
            self.test_loader, self.encoder, self.decoder)

        df_pred = pd.DataFrame(data={'predicted': predicted},
                               index=pd.to_datetime(timeindex).tz_localize('UTC'))
        return df_pred


class ChallengeTimeserieFcst_DA_RNN(TimeserieFcst_DA_RNN):

    def set_train_loaders(self, target_col, train_csvpath,
                          ts_col_label='timestamp', sep=',',
                          cols_to_drop=[],
                          sampler=SubsetRandomSampler,
                          perc_train_valid=0.1,
                          debug=False,
                          ):

        limit_load = 100 if debug else None
        if limit_load:
            logger.info('limit_load set to {} when reading {}.'.format(
                limit_load, train_csvpath))

        with Timer('Reading {}'.format(train_csvpath)):
            df = pd.read_csv(train_csvpath, nrows=limit_load, sep=sep)
            df[ts_col_label] = pd.to_datetime(df[ts_col_label])
            df = df.set_index(ts_col_label).tz_localize('UTC')
            df = df.drop(columns=cols_to_drop)
            df = df.sort_index()

        logger.info("Shape of data: {}.".format(df.shape))
        # logger.info("Missing datas: \n{}.".format(df.isna().sum()))

        # TODO: better NaN handle:
        df = df.fillna(0)

        return super().set_train_loaders(
            df=df, target_col=target_col,
            sampler=sampler, perc_train_valid=perc_train_valid)
