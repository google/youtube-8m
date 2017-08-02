#!/usr/bin/env python

# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import uuid
import os
import requests

URL = os.environ.get('GUESTBOOK_URL')


def test_e2e():
    assert URL
    print ("Running test against {}".format(URL))
    r = requests.get(URL)
    assert b'Guestbook' in r.content
    u = uuid.uuid4()
    data = {'content': str(u)}
    r = requests.post(URL + '/sign', data)
    assert r.status_code == 200
    r = requests.get(URL)
    assert str(u).encode('utf-8') in r.content
    print("Success")

if __name__ == "__main__":
    test_e2e()
