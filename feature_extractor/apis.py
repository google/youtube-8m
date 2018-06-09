from datetime import datetime, timedelta
import json
import os
import threading

from googleapiclient import discovery
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload
from httplib2 import Http
from httplib2 import HttpLib2Error

from cached_property import cached_property
import hashlib
import requests
from httplib2 import Http
from oauth2client.service_account import ServiceAccountCredentials


DAILYMOTION_API = "https://api.dailymotion.com"

VIDEO_API = DAILYMOTION_API + '/video/%(xid)s?fields=%(fields)s'
VIDEO_POST_API = DAILYMOTION_API + '/video/%(xid)s'
TOKEN_API = DAILYMOTION_API + '/oauth/token'
VIDEO_TOPIC_API = (
    DAILYMOTION_API + '/video/%(xid)s/topics?fields=%(fields)s&'
    'limit=%(limit)s&page=%(page)s')
TOPIC_API = DAILYMOTION_API + '/topics'
TOPIC_VIDEO_API = (
    DAILYMOTION_API + '/topic/%(dm_entity_id)s/videos/%(video_id)s')
TOPIC_VIDEO_LIST = (
    DAILYMOTION_API + '/topic/%(topic_xid)s/videos?fields=%(fields)s&'
    'limit=%(limit)s&page=%(page)s'
)
USER_API = DAILYMOTION_API + '/user/%(user_xid)s'

ARTIST_API = DAILYMOTION_API + '/artist/%(artist_xid)s'


dailymotion_cred = {
    "client_id": "6e04fc77e526fc52c3b8",
    "client_secret": "75a08393bb5fd03a6cec30e9a8a9037b8012d918",
}


class BaseClient(object):
    """Base class of web api client."""

    _clients = {}
    session = requests.Session()

    @classmethod
    def get_client(cls, **kwargs):
        _id = '_'.join([
            str(os.getpid()), str(threading.get_ident()),
            str(cls.__qualname__), str(kwargs)
        ])
        if _id not in cls._clients:
            cls._clients[_id] = cls(**kwargs)
        return cls._clients[_id]

    def __init__(self, **kwargs):
        super(BaseClient, self).__init__(**kwargs)


class RESTClient(BaseClient):
    """Rest api client with mongo cache"""

    @classmethod
    def data_hash(cls, payload):
        return hashlib.md5(
            str(payload).encode('utf-8')
        ).hexdigest()

    def __init__(self, **kwargs):
        super(RESTClient, self).__init__(**kwargs)

    def _is_error(self, response):
        raise NotImplementedError('Should return is_error or not')

    def _to_response(self, content):
        raise NotImplementedError(
            'Should transforme text content to api response.')

    def _headers(self):
        return {}

    def get(self, url, encoding=None, **kwargs):
        response = self.session.get(url, headers=self._headers())
        if encoding:
            response.encoding = encoding
        response.raise_for_status()
        api_resp = self._to_response(response.text)
        if self._is_error(api_resp):
            raise RuntimeError(
                "Response error of url %s:\n%s" % (
                    url, response.text)
            )
        return api_resp

    def post(self, url, data, **kwargs):
        response = self.session.post(
            url, headers=self._headers(), data=data)
        response.raise_for_status()
        api_resp = self._to_response(response.text)
        if self._is_error(api_resp):
            self.drop_cache(url)
            raise RuntimeError(
                "Response error of url %s:\n%s" % (
                    url, response.text)
            )
        return api_resp


class JsonRESTClient(RESTClient):
    def _to_response(self, content):
        return json.loads(content)


class DailymotionAPI(JsonRESTClient):
    _jwt_token = None
    _expired_at = None

    def _is_error(self, response):
        return False

    def _headers(self):
        return {
            "Authorization": "Bearer %s" % self.jwt_token,
            "Cache-Control": "no-cache",
        }

    def __init__(self, **kwargs):
        """Init client with client credentials."""
        self._client_id = dailymotion_cred['client_id']
        self._client_secret = dailymotion_cred['client_secret']
        super(DailymotionAPI, self).__init__(**kwargs)

    def batch_get(self, queries, with_cache=True):
        return self.post(DAILYMOTION_API, data=queries, with_cache=with_cache)

    @property
    def jwt_token(self):
        """Access token of dailymotion api."""
        if self._jwt_token and self._expired_at > datetime.utcnow():
            return self._jwt_token

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Cache-Control": "no-cache",
        }
        data = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "grant_type": "client_credentials",
        }
        resp = self.session.post(TOKEN_API, headers=headers, data=data).json()

        if 'access_token' in resp:
            self._jwt_token = resp['access_token']
            self._expired_at = datetime.utcnow() + timedelta(
                seconds=resp['expires_in'])
            return self._jwt_token
        else:
            raise RuntimeError(resp)

    def video_h264_url(self, xid):
        return self.get_video_meta(xid, 'stream_h264_ld_url')

    def video_title(self, xid):
        return self.get_video_meta(xid, 'title')

    def video_user_id(self, xid):
        return self.get_video_meta(xid, 'user.id')

    def video_tags(self, xid):
        return self.get_video_meta(xid, 'tags')

    def video_description(self, xid):
        return self.get_video_meta(xid, 'description')

    def video_detected_language(self, xid):
        return self.get_video_meta(xid, 'detected_language')

    def wmid(self, wmid, with_cache=False):
        url = DAILYMOTION_API + '/video/UMG:%s' % wmid
        return self.get(
            url, with_cache=with_cache, expired_at=datetime.utcnow() + timedelta(days=30))

    def get_video_meta(self, xid, field, with_cache=True):
        """."""
        value = self.video_metas(xid, [field, ], with_cache).get(field)
        return value

    def video_metas(self, xid, fields, with_cache=True):
        url = VIDEO_API % {
            'xid': xid,
            'fields': ','.join(fields)
        }
        return self.get(
            url, expired_at=datetime.utcnow() + timedelta(days=30), with_cache=with_cache)

    def add_music_meta(self, xid, **fields):
        url = VIDEO_POST_API % {
            'xid': xid,
        }
        return self.post(
            url, fields)

    def _json_request(self, xid, fields):
        return {
            "call": "GET /video/%s" % xid,
            "args": {
                "fields": fields,
            },
        }

    def video_metas_by_batch(self, xids, fields, with_cache=False):
        data = [self._json_request(xid, fields) for xid in xids]
        return self.post(
            DAILYMOTION_API,
            data,
            with_cache=with_cache
        )


class GoogleClient(BaseClient):
    scopes = None
    credential_json = 'google_cred.json'

    @cached_property
    def credentials(self):
        return ServiceAccountCredentials.from_json_keyfile_name(
            self.credential_json, self.scopes)

    @cached_property
    def cred_http(self):
        http = Http()
        self.credentials.authorize(http)
        return http


class GoogleCloudStorageClient(GoogleClient):
    """Google cloud storage api."""

    scopes = ['https://www.googleapis.com/auth/devstorage.full_control']

    def __init__(self):
        """Init GCS with credentials.

        :param credentials: google api credential.
        :type credentials: oauth2client.client.Credentials
        """
        self.service = discovery.build('storage', 'v1', http=self.cred_http)
        self.chunk_size = 2 * 1024 * 1024
        self.max_retry = 5
        super(GoogleCloudStorageClient, self).__init__()

    def get_acl(self, bucket, name):
        try:
            return self.service.objectAccessControls().list(
                bucket=bucket, object=name).execute()
        except HttpError as err:
            if err.resp.status == 404:
                return None
            else:
                raise

    def is_public(self, bucket, name):
        acs = self.get_acl(bucket, name)
        if acs is None:
            raise RuntimeError("%s is not in bucket %s" % (name, bucket))
        for ac in acs.get('items', []):
            if ac['entity'] == 'allUsers' and ac['role'] == 'READER':
                return True
        return False

    def update_acl(self, bucket, name, entity, body):
        try:
            return self.service.objectAccessControls().update(
                bucket=bucket, object=name, entity=entity, body=body).execute()
        except HttpError as err:
            if err.resp.status == 404:
                return None
            else:
                raise

    def enable_public_access(self, bucket, name):
        return self.update_acl(
            bucket, name, 'allUsers', {'role': 'READER'})

    def get_media(self, bucket, name):
        """Get the content of the GCS object.

        :param bucket: bucket name.
        :type bucket: str
        :param name: object name.
        :type name: str
        :returns: object content.
        :rtype: bytes
        """
        try:
            return self.service.objects().get_media(
                bucket=bucket, object=name).execute()
        except HttpError as err:
            if err.resp.status == 404:
                return None
            else:
                raise

    def get_meta(self, bucket, name):
        """Test object exist on GCS.

        :param bucket: bucket name.
        :type bucket: str
        :param name: object name.
        :type name: str
        :returns: object metadata.
        :rtype: dict
        """
        try:
            return self.service.objects().get(
                bucket=bucket, object=name).execute()
        except HttpError as err:
            if err.resp.status == 404:
                return None
            else:
                raise err

    def list(self, bucket, name, delimiter="/"):
        try:
            return self.service.objects().list(
                bucket=bucket, prefix=name, delimiter="/").execute()
        except HttpError as err:
            if err.resp.status == 404:
                return None
            else:
                raise err

    def is_exist(self, bucket, name):
        """Test if object already exist in bucket."""
        # TODO: compare file md5 hash
        return self.get_meta(bucket, name) is not None

    def get_or_update(self, bucket, name, bytes_stream,
                      mimetype, chunk_size=None):
        """Get object link, otherwise create object.

        :param bucket: bucket name.
        :type bucket: str
        :param name: object name.
        :type name: str
        :param bytes_stream: bytes stream.
        :type bytes_stream: stream
        :param mimetype: mime type.
        :type mimetype: str
        :param chunk_size chunk size for a request.
        :type chunk_size: long
        """
        previous_meta = self.get_meta(bucket, name)
        if previous_meta:
            return previous_meta
        update_info = self.update(
            bucket, name, bytes_stream, mimetype, chunk_size)
        return update_info

    def update(self, bucket, name, bytes_stream, mimetype, chunk_size=None):
        """Update content from a bytes stream to GCS.

        :param bucket: bucket name.
        :type bucket: str
        :param name: object name.
        :type name: str
        :param bytes_stream: bytes stream.
        :type bytes_stream: stream
        :param mimetype: mime type.
        :type mimetype: str
        :param chunk_size chunk size for a request.
        :type chunk_size: long
        """
        media = MediaIoBaseUpload(
            bytes_stream,
            mimetype=mimetype,
            chunksize=self.chunk_size,
            resumable=True
        )
        request = self.service.objects().insert(
            bucket=bucket, name=name, media_body=media)

        progressless_iters = 0
        response = None
        self._debug('uploading')
        while response is None:
            error = None
            try:
                progress, response = request.next_chunk()
                if progress:
                    self._debug(
                        'Upload %d%%' % (100 * progress.progress()))
            except HttpError as err:
                error = err
                if err.resp.status < 500:
                    raise
            except RETRYABLE_ERRORS as err:
                error = err

            if error:
                progressless_iters += 1
                self._handle_error(error, progressless_iters)
            else:
                progressless_iters = 0
        self._debug('upload finished')
        return response

    def _handle_error(self, error, progressless_iters):
        if progressless_iters > self.max_retry:
            self._error(
                'Failed to make progress for too many consecutive iterations.')
        raise error

        sleeptime = 2 * progressless_iters
        self._error(
            'Caught exception (%s). Sleeping for %s seconds before retry #%d.'
            % (str(error), sleeptime, progressless_iters)
        )
        time.sleep(sleeptime)
