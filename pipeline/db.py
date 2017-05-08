import psycopg2
import itertools

from itertools import groupby
from operator import itemgetter
from heapq import merge

def fetch(host, dbname, user, password, query):
    '''
    Executes query on a given database
    '''
    connection_string = "dbname='{0}' user='{1}' host='{2}' password='{3}'".format(
        dbname, user, host, password)
    try:
        conn = psycopg2.connect(connection_string)
        curr = conn.cursor()
        curr.execute(query)
        res = curr.fetchall()
    except Exception as e:
        print(e)
        logging.exception('')
    finally:
        curr.close()
        conn.close()

    return res

def inner_join(a, b):
    '''
    Joins two iterables of tuples on the first
    element

    Arguments:
    a - list of tuples (id, x)
    b - list of tuples (id, y)

    Returns:
    list of tuples (id, x, y)
    '''
    key = itemgetter(0)
    a.sort(key=key) 
    b.sort(key=key)
    for _, group in groupby(merge(a, b, key=key), key):
        row_a, row_b = next(group), next(group, None)
        if row_b is not None: # join
            yield row_a + row_b[1:]


def filter_videos(videos, min_count = 10):
    '''
    Filters videos and returns mapping to the original tags

    Returns:
    filtered       - a list with transformed and filtered videos
    tags_2_indices - a dictinary that transforms tags to rank of their
                     frequencies
    indices_2_tags - inverse dictionary
    '''

    # we have to iterate twice, first to create dictionary, then 
    # then to filter tags and transform the list
    if not isinstance(videos, list):
        videos = list(videos)

    # Filters top tags and creates mapping
    count = Counter(itertools.chain(*[tup[1] for tup in videos]))
    tags_2_indices = {
        tag_id: index
            for index, (tag_id, count) in enumerate(count.most_common(), 1)
            if count >= min_count
    }

    # reverse index for decoding 
    indices_2_tags = {
        v: k for k, v in tags_2_indices.items()
    }

    filtered = []
    for video_id, tags, url in videos:
        encoded = [tags_2_indices[t] for t in tags if t in tags_2_indices]
        if encoded:
            filtered.append((video_id, encoded, url))

    return filtered, tags_2_indices, indices_2_tags
