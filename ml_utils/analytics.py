from collections import namedtuple
import pandas as pd
from pandas.api.types import is_datetimetz
from pandas.core.algorithms import mode
from pandas.tseries.frequencies import to_offset


def tz_convert_multiindex(ts, tz='UTC'):
    """Convert all aware indexes of multiIndex timeserie.
    It also checks first if the indexes are effectively aware.
    """
    for i in range(len(ts.index.levels)):
        assert is_datetimetz(ts.index.levels[i])
        ts.index = ts.index.set_levels(ts.index.levels[i].tz_convert('UTC'),
                                       level=i)
    return ts


def tz_localize_multiindex(ts, tz='UTC'):
    """Localize all naive indexes of multiIndex timeserie.
    It also checks first if the indexes are effectively naives.
    """
    for i in range(len(ts.index.levels)):
        assert not is_datetimetz(ts.index.levels[i])
        ts.index = ts.index.set_levels(ts.index.levels[i].tz_localize('UTC'),
                                       level=i)
    return ts


def detect_frequency(idx):
    """
    Return the most plausible frequency of DatetimeIndex idx (even when gaps in it).
    It calculates the delta between element of the index (idx[1:] - idx[:1]), gets the 'mode' of the delta (most frequent delta) and transforms it into a frequency ('H','15T',...)

    A solution exists in pandas:
    ..ipython:
        from pandas.tseries.frequencies import _TimedeltaFrequencyInferer
        inferer = _TimedeltaFrequencyInferer(idx)
        freq = inferer.get_freq()

    But for intraday frequencies, if it is not regular (like for 'publication_date'
    of forecast timeseries), then the inferer.get_freq() return None.
    In those cases, we are going to return the smallest frequency possible.

    :param idx: DatetimeIndex
    :return: str
    """
    if len(idx) < 2:
        raise ValueError(
            "Cannot detect frequency of index when index as less than two elements")

    # calculates the delta
    delta_idx = idx[1:] - idx[:-1]
    delta_mode = mode(delta_idx)

    if len(delta_mode) == 0:
        # if no clear mode, take the smallest delta_idx
        td = min(delta_idx)
    else:
        # infer frequency from most frequent timedelta
        td = delta_mode[0]

    return to_offset(td)


TSAnalytics = namedtuple("TSAnalytics",
                         "freq sorted continuous gaps duplicates")


def analyse_datetimeindex(idx, start_date=None, end_date=None, freq=None):
    """Check if the given index is of type DatetimeIndex & is aware.
    Returns the implied frequency, a sorted flag, the list of continuous segment, the list of gap segments and the list of duplicated indices.
    Continuous and gaps segments are expressed as [start:end] (both side inclusive).
    If the index is not sorted, it will be sorted before checking for continuity.
    Specifying start_date and end_date check for gaps at beginning and end of the index.
    Specifying freq enforces control of gaps according to frequency.


    :param idx: DatetimeIndex aware
    :param start_date: datetime expression
    :param end_date: datetime expression
    :param freq: str
    :return: a named tuple with (freq, sorted, continuous, gaps, duplicates)
    """
    assert isinstance(idx, pd.DatetimeIndex)

    if not is_datetimetz(idx):
        raise ValueError("Naive DatetimeIndex is forbidden for your own sake."
                         "idx={}".format(idx))

    if len(idx) < 2:
        return TSAnalytics(None, True, [], [], [])

    if start_date is None:
        start_date = idx[0]
    else:
        start_date = pd.Timestamp(start_date)

    if end_date is None:
        end_date = idx[-1]
    else:
        end_date = pd.Timestamp(end_date)

    if not is_datetimetz(pd.DatetimeIndex([start_date, end_date])):
        raise ValueError("One of the following date is not aware:\n"
                         "start_date={}\nend_date={}".format(
                             start_date, end_date))

    if freq is None:
        freq = detect_frequency(idx)

    if not idx.is_unique:
        duplicates_flag = idx.duplicated(keep="first")
        duplicates = idx[duplicates_flag].tolist()
        idx = idx[~duplicates_flag]
    else:
        duplicates = []

    sorted = idx.is_monotonic_increasing

    idx_full = pd.date_range(
        start=start_date, end=end_date, tz=idx.tz, freq=freq)
    sr_full = pd.Series(index=idx, data=1).reindex(idx_full, fill_value=0)
    sr_shift = sr_full.diff(1)

    # detect first item in start, stop
    first_changes = sr_full[sr_shift != 0.]
    last_changes = sr_full[sr_shift.shift(-1) != 0.]
    # stops = sr_full[sr_shift == -1.]
    assert len(first_changes) == len(last_changes)
    segments = {0: [], 1: []}
    for (ts, modes), (te, modee) in zip(first_changes.iteritems(), last_changes.iteritems()):
        assert modes == modee
        segments[modes].append((ts, te))

    return TSAnalytics(freq, sorted, segments[1], segments[0], duplicates)
