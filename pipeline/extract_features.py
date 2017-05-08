MODEL_PATH = '/models/image/inception/classify_image_graph_def.pb'
DATA_PATH = '/data/video/video-level-features/'
NPROD = 4
LIMIT = 10
MIN_TAGS = 10
LOGGING_INTERVAL = 2

VQUERY = "select post_id, url from videos where status='ok'"
TQUERY = "select id, tags from videos where tags is not NULL"
TAGS = "select tag_id, name, path from content_tags"

import logging
import os
import time

import mp
import sq

from db import fetch, inner_join, filter_videos
from utils.logging import setup_logging


def run_and_measure(fun, n):
    logger = logging.getLogger(__name__)
    start_time = time.time()
    fun()
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info("Elapsed time was %g seconds [%g]" % (elapsed, elapsed/n))


if __name__ == '__main__':
    setup_logging()
    logger = logging.getLogger(__name__)

    host='192.95.32.117'

    vdbname='ds-wizards'
    vuser='wizard'
    vpassword='GaG23jVxZhMnQaU53r8o'

    tdbname='ds-content-tags'
    tuser='ds-content-tags'
    tpassword='0fXjWl592vNf1gYvIw8w'

    vres = fetch(host, vdbname, vuser, vpassword, VQUERY)
    vres = [(post_id.split("_")[1], url) for post_id, url in vres]

    tres = fetch(host, tdbname, tuser, tpassword, TQUERY)
    videos = inner_join(tres, vres)

    filtered, t2i, i2t = filter_videos(videos, MIN_TAGS)
    logger.info("Found %d videos with %d unique tags" % (len(filtered), len(t2i)))

    # we will need thid eventually
    tags = {
	tag_id: (name, path) for (tag_id, name, path) in fetch(
            host, tdbname, tuser, tpassword, TAGS)
    }


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    work = filtered[:LIMIT]

    fsq = lambda : sq.fetch(work,
                            model_path=MODEL_PATH,
                            data_path=os.path.join(DATA_PATH, 'seq'),
                            logging_step=LOGGING_INTERVAL)

    fmp = lambda : mp.fetch(work, nprod=NPROD,
                            model_path=MODEL_PATH,
                            data_path=os.path.join(DATA_PATH, 'mp'),
                            logging_step=LOGGING_INTERVAL)

    run_and_measure(fmp, len(work))
