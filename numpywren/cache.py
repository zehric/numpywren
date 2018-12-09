#
# Copyright 2018 PyWren Team
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
#

from __future__ import absolute_import


import collections
import os
import mmap
import tempfile

import boto3

TEMP_DIR = tempfile.gettempdir()

class Entry(object):

    def __init__(self, key, storage=None):
        self.ref_cnt = 0
        self.dirty = False
        self.file_path = os.path.join(TEMP_DIR, key)
        if storage:
            storage.load(key, self)
        self.fd = os.open(self.file_path, os.O_RDWR)
        self.data = mmap.mmap(self.fd, 0)

    def close(self):
        self.data.close()
        os.ftruncate(self.fd, 0)
        os.close(self.fd)
        os.unlink(self.file_path)


class Cache(object):
    """
    A Cache object caches data from S3.
    """
    # TODO: separate notion of S3 keys from file name
    def __init__(self, config):
        self.storage = Storage(config)
        self.cache = collections.OrderedDict()
        self.max_entries = 1 # TODO: to change this

    def release(self, key):
        """
        Decrements the reference counter for the cache entry.
        :param key: data key
        :return: bool success
        """
        entry = self.cache.pop(key)
        entry.ref_cnt -= 1
        self.cache[key] = entry
        return True

    def evict_if_full(self):
        evict_key = None
        if len(self.cache) >= self.max_entries:
            for key, entry in self.cache.items():
                if entry.ref_cnt == 0:
                    evict_key = key
                    break
        if evict_key:
            entry = self.cache.pop(evict_key)
            if entry.dirty:
                self.storage.store(key, entry)
            entry.close()
        # TODO: do something if full

    def get(self, key):
        """
        Requests key from the cache. If it does not exist, request the data
        from the backing store. Pins the cache entry as well.
        :param key: data key
        :return: str path to shared data
        """
        if key in self.cache:
            entry = self.cache.pop(key)
            self.cache[key] = entry
        else:
            entry = Entry(key, storage=self.storage)
            self.evict_if_full()
            self.cache[key] = entry

        entry.ref_cnt += 1
        return entry.file_path

    def put(self, key):
        """
        Marks data as dirty. On eviction, the cache writes the data to
        the backing store. Does not pin the cache entry, but this counts
        as an access.
        :param key: data key
        :return: bool success
        """
        if key in self.cache:
            entry = self.cache.pop(key)
            self.cache[key] = entry
        else:
            self.evict_if_full()
            entry = Entry(key)
            self.cache[key] = entry

        entry.dirty = True
        return True

class Storage(object):
    """
    A Storage object interfaces between the cache and the backing store.
    """

    def __init__(self, s3_bucket):
        self.s3_bucket = s3_bucket
        self.s3client = boto3.client('s3')

    def load(self, key, cache_entry):
        self.s3client.download_file(self.s3_bucket, key, cache_entry.file_path)

    def store(self, key, cache_entry):
        self.s3client.upload_fileobj(cache_entry.data, self.s3_bucket, key)
