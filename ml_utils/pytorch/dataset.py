import logging
import pandas as pd
import PIL
import random
from math import floor

from sklearn.preprocessing import LabelBinarizer
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch import from_numpy, np

logger = logging.getLogger()


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
        ids_missing_mask = []
        for i, row in df.reset_index().iterrows():
            fpath = img_path / (str(int(row['id'])) + img_ext)
            ids_missing_mask.append(fpath.exists())

        if not all(ids_missing_mask):
            raise ValueError("Some images referenced in the CSV file where not "
                             "found: {}".format(
                                 df['id'][[not i for i in ids_missing_mask]]))

        self.df = df
        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.ids = self.df.index

        self.lb = LabelBinarizer()
        if 'orientation' in self.df.columns.tolist():
            self.labels = self.df['orientation'].values - 1
            self.labels_binarized = self.lb.fit_transform(self.df['orientation']).astype(np.float32)
        else:
            self.labels = np.zeros(self.df.shape[0])
            self.labels_binarized = np.zeros(self.df.shape)

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
