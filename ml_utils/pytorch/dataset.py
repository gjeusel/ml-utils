import logging
import pandas as pd
import PIL
import random
from math import floor

import numpy as np

from sklearn.preprocessing import LabelBinarizer
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch

from .profiling import Timer

logger = logging.getLogger()


###   SHOULD LOOK OUT FOR ImageFolder in Torchvision ...
class CustomDataset(Dataset):
    """Dataset wrapping images and target labels.
    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        optional: A torchvision transforms
        optional: a limit of images to take into account
    """

    def __init__(self, csv_path, img_path, img_ext='.jpg',
                 transform=transforms.Compose(
                     [transforms.Resize(size=(64, 64)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                           std=(0.5, 0.5, 0.5))]),
                 limit_load=None):

        logger.info('Initializing dataset using {} with {}'.format(
            csv_path.as_posix(),
            [t.__class__.__name__ for t in transform.transforms])
        )
        if limit_load:
            logger.info('limit_load set to {}'.format(limit_load))

        df = pd.read_csv(csv_path, index_col=0, nrows=limit_load)
        df.index.name = 'id'
        assert len(df.columns) == 1
        df.columns = ['label']
        ids_missing_mask = []
        for i, row in df.reset_index().iterrows():
            fpath = img_path / (str(int(row['id'])) + img_ext)
            ids_missing_mask.append(fpath.exists())

        if not all(ids_missing_mask):
            raise ValueError("Some images referenced in the CSV file where not "
                             "found: {}".format(
                                 df.index[[not i for i in ids_missing_mask]]))

        self.df = df
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.ids = self.df.index

        self.lb = LabelBinarizer()
        self.labels = self.df['label'].values
        try:
            self.labels_binarized = self.lb.fit_transform(self.df['label']).astype(np.float32)
        except ValueError as e:
            logger.debug(e)

    def __getitem__(self, index):
        """Return data at index."""
        fname = self.img_path / (str(self.ids[index]) + self.img_ext)
        img = PIL.Image.open(fname)
        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.df.index)

    def getLabelEncoder(self):
        return self.lb


def train_valid_split(dataset, test_size=0.25, shuffle=False, random_seed=0):
    """ Return a list of splitted indices from a DataSet.
    Indices can be used with DataLoader to build a train and validation set.

    Arguments:
        A Dataset
        A test_size, as a float between 0 and 1 (percentage split) or as an int (fixed number split)
        Shuffling True or False
        Random seed
    """
    logger.info('Splitting dataset with test_size={}%'.format(test_size*100))
    length = len(dataset)
    indices = list(range(1, length))

    if shuffle:
        random.seed(random_seed)
        random.shuffle(indices)

    if type(test_size) is float:
        split = floor(test_size * length)
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or a float' % str)
    return indices[split:], indices[:split]


class TsFcstDataset(Dataset):
    def __init__(self, n_timestep, df, target_col, test_mode=False):

        self.n_timestep = n_timestep
        self.df = df
        self.test_mode = test_mode

        self.X = df[df.columns.difference([target_col])].values
        self.y = df[target_col].values

        assert pd.api.types.is_datetimetz(df.index)
        self.df = df.tz_convert('UTC')
        self.timeindex = self.df.index.astype(np.int64).values

    def __getitem__(self, index):
        """Return data at index."""
        X_batch = self.X[index: (index + self.n_timestep - 1), :]
        y_history = self.y[index: (index + self.n_timestep - 1)]

        # Timeindex of predicted register:
        timeindex = self.timeindex[index + self.n_timestep]

        # Convert to FloatTensor:
        X_batch = torch.FloatTensor(X_batch)
        y_history = torch.FloatTensor(y_history)

        if not self.test_mode:
            y_target = self.y[index + self.n_timestep]
            y_target = torch.FloatTensor([y_target])
        else:
            y_target = []

        return X_batch, y_history, y_target, timeindex

    def __len__(self):
        return len(self.df.index) - self.n_timestep
