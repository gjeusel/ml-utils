from timeit import default_timer as timer
import logging
logger = logging.getLogger(__name__)


class Timer:

    def __init__(self, label, enterlog=False):
        self.label = label
        self.enterlog = enterlog

    def __enter__(self):
        if self.enterlog:
            logger.info('{} in progress...'.format(self.label))
        self.start = timer()
        return self

    def __exit__(self, *args):
        self.end = timer()
        self.interval = self.end - self.start
        logger.info('%s took %.03f sec', self.label, self.interval)
