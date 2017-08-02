#
## Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

.PHONY: e2e

VERSION=e2e-test

.PHONY: all
all: deploy

.PHONY: deploy
deploy:
	appcfg.py update . -A $(GAE_PROJECT) --version=$(VERSION)

.PHONY: e2e_test
e2e_test: export GUESTBOOK_URL = http://$(VERSION)-dot-$(GAE_PROJECT).appspot.com
e2e_test: deploy
	pip install -r e2e/requirements-dev.txt
	python e2e/test_e2e.py
 
