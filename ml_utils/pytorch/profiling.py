from timeit import default_timer as timer
import logging

logger = logging.getLogger(__file__)


class Timer:
    def __init__(self, label, csvname=None):
        self.label = label

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        self.end = timer()
        self.interval = self.end - self.start
        logger.info('%s took %.03f sec', self.label, self.interval)
