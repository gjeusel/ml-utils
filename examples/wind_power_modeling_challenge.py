import fire
from pathlib import Path
import logging
import pandas as pd
from tensorboardX import SummaryWriter

from flask import Flask
from sklearn import preprocessing, model_selection, metrics

from ml_utils import dashboard
from ml_utils.profiling import Timer
from ml_utils.pytorch import TimeserieFcst_DA_RNN

import lightgbm as lgb
from lightgbm import LGBMRegressor


logger = logging.getLogger(__name__)


root_dir = Path(__file__).parent
data_dir = root_dir / 'data/'
snapshot_dir = root_dir / 'snapshots/'
sub_dir = root_dir / 'submission/'

tb_writer = SummaryWriter()

params_best_fit = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2_root'},

    'learning_rate': 0.08,
    'max_depth': 30,
    'n_estimators': 300,
    'num_leaves': 1400,

    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
}


def datetime_features(df):
    idx = df.index
    df['year'] = idx.year
    df['month'] = idx.month
    df['day'] = idx.day
    df['dayofyear'] = idx.dayofyear
    df['hour'] = idx.hour
    df['minute'] = idx.minute

    df['week'] = idx.week
    df['weekday'] = idx.weekday
    return df


class WindPowerChallenge():
    train_csvpath = data_dir / 'train.CSV'
    test_csvpath = data_dir / 'test.CSV'

    def __init__(self, debug=False):
        self.debug = debug

    def _read_csv(self, csv):
        if self.debug:
            limitrows = 100
        else:
            limitrows = None

        if csv == 'test':
            path = self.test_csvpath
        elif csv == 'train':
            path = self.train_csvpath
        else:
            raise ValueError('Wrong csv')

        df = pd.read_csv(path, sep=';', nrows=limitrows)
        df = df.set_index('Date')
        df.index = pd.to_datetime(df.index, utc=True)

        # Label Encoder:
        le_maccode = preprocessing.LabelEncoder()
        le_maccode.fit(df['Mac_Code'])
        # print(le_maccode.classes_)
        df['Mac_Code'] = le_maccode.transform(df['Mac_Code'])

        self.le_maccode = le_maccode

        return df

    def dash(self, csv='train'):
        with Timer('Reading {} csv'.format(csv), True):
            dftrain = self._read_csv(csv)

        server = Flask(__name__)
        dashboard.register_dashboard(df=dftrain, server=server)

        server.run(debug=True)

    def best_params_discovery(self):
        with Timer('Processing train csv', True):
            df = WindPowerChallenge._read_csv()
            # Drop nans for the moment
            df = df.dropna()

        X_train = df.drop('TARGET', axis=1)
        y_train = df['TARGET']

        param_dist = {
            'learning_rate': [0.08, 0.1, 0.15],
            'max_depth': [20, 25, 30],
            'num_leaves': [1000, 1200, 1400],
            'n_estimators': [100, 200, 300],
            # 'min_data_in_leaf': [800],
        }

        model = lgb.LGBMRegressor(silent=False)

        # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
        with Timer('Randomized Search Cross Validation', True):
            rand_grid_search = model_selection.RandomizedSearchCV(
                model, param_distributions=param_dist,
                n_iter=30)
            rand_grid_search.fit(X_train, y_train.values.ravel())

        # # http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        # with Timer('Grid Search Cross Validation', True):
        #     grid_search = model_selection.GridSearchCV(
        #         model, param_grid=param_dist, scoring='neg_mean_squared_error',
        #         cv=3, verbose=10,
        #         # n_jobs=4,  # not working
        #     )
        #     grid_search.fit(X_train, y_train.values.ravel())

        # Found:
        # {'learning_rate': 0.08,
        #  'max_depth': 30,
        #  'n_estimators': 300,
        #  'num_leaves': 1400}
        return rand_grid_search

    def playing_around(self):
        with Timer('Reading train csv', True):
            dftrain = self._read_csv('train')

        dftrain = datetime_features(dftrain)

        with Timer('Splitting train-test set'):
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                dftrain.drop('TARGET', axis=1),
                dftrain[['TARGET']],
                test_size=0.2, random_state=42)

        # Normalize ?:
        # # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        # sc = preprocessing.StandardScaler()
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.fit_transform(y_test)

        d_train = lgb.Dataset(X_train, y_train.values.ravel())
        d_valid = lgb.Dataset(X_test, y_test.values.ravel(), reference=d_train)

        with Timer('Training lightgbm model', True):
            gbm = lgb.train(params_best_fit, d_train, valid_sets=d_valid,
                            num_boost_round=20,
                            early_stopping_rounds=5,
                            )

        __import__('IPython').embed()  # Enter Ipython
        # with Timer('Predicting using lightgbm model', True):
        #     y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

        # rmse = metrics.mean_squared_error(y_test, y_pred) ** 0.5
        # logger.info('Validation RMSE: {}'.format(rmse))

    def predict_submit(self):
        with Timer('Reading train csv', True):
            dftrain = self._read_csv('train')
            dftrain = datetime_features(dftrain)

        X_train = dftrain.drop('TARGET', axis=1)
        y_train = dftrain['TARGET']

        d_train = lgb.Dataset(X_train, y_train.values.ravel())

        with Timer('Training lightgbm model', True):
            gbm = lgb.train(params_best_fit, d_train,
                            num_boost_round=20,
                            )

        with Timer('Reading test csv', True):
            dftest = self._read_csv('test')
            dftest = datetime_features(dftest)

        with Timer('Predicting using lightgbm model', True):
            y_pred = gbm.predict(dftest, num_iteration=gbm.best_iteration)

        with Timer('Saving csv'):
            now = pd.Timestamp.now(tz='CET').tz_localize(None)
            fname = 'predicted_lightgbm_{}.csv'.format(now.strftime('%Y.%m.%d-%Hh%M'))
            dfres = pd.DataFrame(data=y_pred, columns=['pred'])
            dfres.to_csv('data/{}'.format(fname), index=False)

    # Neural Network:
    # da_rnn = TimeserieFcst_DA_RNN(
    #     n_timestep=10, snapshot_dir=snapshot_dir, sub_dir=sub_dir,
    #     batch_size=4, num_workers=4)
    # da_rnn.set_train_loaders(df, target_col='TARGET')
    # da_rnn.set_encoder_decoder()
    # da_rnn.train()


if __name__ == '__main__':
    import sys
    log_root = logging.getLogger()
    log_root.setLevel(logging.DEBUG)

    # define a Handler which writes INFO messages or higher to the sys.stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('|%(name)s|%(levelname)s|: %(message)s')
    ch.setFormatter(formatter)

    # create file handler which logs even debug messages
    fh = logging.FileHandler((root_dir / 'dash.log').as_posix())
    fh.setLevel(logging.DEBUG)

    # add the handler to the root logger
    log_root.addHandler(ch)
    log_root.addHandler(fh)

    # Del first one that was automatically generated:
    del log_root.handlers[0]

    wp = WindPowerChallenge()
    # wp.playing_around()

    fire.Fire({
        'wind': WindPowerChallenge,
    })
