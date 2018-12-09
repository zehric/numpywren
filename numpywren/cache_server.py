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

from multiprocessing.connection import Listener
import sys
import os
from threading import Thread
import click
from cache import Cache

class CacheServer(object):
    def __init__(self, s3_bucket):
        self._cache = Cache(s3_bucket)
        self._socket_listener = Listener(address='local_cache')
    def get(self, k):
        return self._cache.get(k)
    def put(self, k):
        return self._cache.put(k)
    def release(self, k):
        return self._cache.release(k)
    def _listen(self):
        while True:
            try:
                client = self._socket_listener.accept()
                thread = Thread(target=self._serve_client, args=(client,))
                thread.start()
            except:
                # TODO: throw error
                self._socket_listener.close()
                sys.exit(1)
    def _serve_client(self, client):
        while True:
            client.poll()

            try:
                request = client.recv()
            except EOFError:
                return
            else:
                if request[0] == 0: # get
                    client.send(self.get(request[1]))
                elif request[0] == 1: # put
                    client.send(self.put(request[1]))
                elif request[0] == 2: # release
                    client.send(self.release(request[1]))
                elif request[0] == 'exit':
                    self._socket_listener.close()
                    os._exit(1)

def cache_run(cache):
    cache._listen()

@click.command()
@click.option('--s3_bucket', required=True, help='s3 bucket')
def server(s3_bucket):
    cache = CacheServer(s3_bucket)
    cache._listen()

if __name__ == '__main__':
    server()
