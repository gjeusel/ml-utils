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
from .predict import validate_da_rnn

logger = logging.getLogger(__file__)


class ImageClassification():
    """Wrapper class for image classification."""

    def __init__(self, train_csvpath, train_imgdir, snapshot_dir,
                 test_csvpath=None, test_imgdir=None, sub_dir=None,
                 batch_size=4, num_workers=4,
                 tb_writer=None,
                 ):

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
                          debug=False,
                          ):

        self.ds_transform_augmented = ds_transform_augmented
        self.ds_transform_raw = ds_transform_raw

        # Loading the dataset
        limit_load = 100 if debug else None
        X_train = CustomDataset(self.train_csvpath, self.train_imgdir,
                                transform=ds_transform_augmented,
                                limit_load=limit_load,
                                )
        X_val = CustomDataset(self.train_csvpath, self.train_imgdir,
                              transform=ds_transform_raw,
                              limit_load=limit_load,
                              )

        # Creating a validation split
        train_idx, valid_idx = train_valid_split(X_train, perc_train_valid)

        if sampler is not None:
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
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

    def train(self, epochs, net, loss_func, optimizer,
              score_func=log_loss, score_type='proba'):
        """Train the network."""

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

                if best_score > score and epoch > 4:
                    best_score = score
                    save_snapshot(epoch+1, net, score,
                                  optimizer, self.snapshot_dir)

        self.net = net

    def continue_training(self, epochs, net, loss_func, optimizer, pth_path):
        """Continue training a network."""
        epoch_start, net_state_dict, score, optimizer_state_dict = load_snapshot(
            pth_path)
        assert epochs >= epochs

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

                if best_score < score:
                    best_score = score
                    save_snapshot(epoch+1, net, score, optimizer)

    def predict(self, net, pth_path, ds_transform_raw, score_type='proba',
                batch_size=4, num_workers=4, debug=False):

        limit_load = 100 if debug else None
        X_test = CustomDataset(self.test_csvpath, self.test_imgdir,
                               transform=ds_transform_raw,
                               limit_load=limit_load,
                               )

        test_loader = DataLoader(X_test,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 # pin_memory=True,
                                 )

        self.X_test, self.test_loader = X_test, test_loader

        # Load net from best iteration
        epoch, net_state_dict, score, _ = load_snapshot(pth_path)
        net.load_state_dict(net_state_dict)

        return predict_image_classification(test_loader, net, score_type=score_type)

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

    def set_train_loaders(self, target_col, train_csvpath,
                          ts_col_label='timestamp', sep=',',
                          cols_to_drop=[],
                          sampler=SubsetRandomSampler,
                          perc_train_valid=0.1,
                          debug=False,
                          ):

        limit_load = 100 if debug else None
        dataset = TsFcstDataset(n_timestep=self.n_timestep, target_col=target_col,
                                csv_path=train_csvpath, ts_col_label=ts_col_label,
                                cols_to_drop=cols_to_drop, sep=sep,
                                limit_load=limit_load)

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
                                  # sampler=train_sampler,
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

    def set_encoder_decoder(self,
                            encoder_hidden_size=64, decoder_hidden_size=64,
                            n_timestep=10,
                            learning_rate=0.01,
                            ):

        self.n_timestep = n_timestep

        self.encoder = encoder(input_size=self.ncols-1,
                               hidden_size=encoder_hidden_size,
                               n_timestep=n_timestep)

        self.decoder = decoder(encoder_hidden_size=encoder_hidden_size,
                               decoder_hidden_size=decoder_hidden_size,
                               n_timestep=n_timestep)

        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)


        self.encoder_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, self.encoder.parameters()),
            lr=learning_rate)
        self.decoder_optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr=learning_rate)

    def train(self, epochs=10, loss_func=nn.MSELoss()):

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

            # if best_score > score and epoch > 4:
            #     best_score = score
            #     save_snapshot(epoch+1, net, score,
            #                     optimizer, self.snapshot_dir)
