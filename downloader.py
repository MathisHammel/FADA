import requests
import logging
import time
import os
from threading import Thread

USER_AGENT = 'fada'
INTERVAL = 1.2

logger = logging.getLogger('app')
logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')

class Downloader(Thread):

    def __init__(self, url, output):
        Thread.__init__(self)
        self.url = url
        self.output = output

    def run(self):
        try:
            req = requests.get(self.url, headers={'User-Agent':USER_AGENT})
            
            if req.status_code != 200:
                raise Exception(r.status_code)
            
            with open(self.output, 'wb') as f:
                f.write(req.content)
            
            req_size = int(req.headers['content-length'])
            file_size = os.path.getsize(self.output)
            
            if file_size != req_size:
                raise Exception('Image is corrupted, size is {} ( expected {} )'.format(file_size, req_size))
        except Exception as e:
            logger.error(e)

if __name__ == '__main__':
    while True:
        output = 'downloads/' + str(int(time.time())) + '.jpeg'
        logger.info('Downloading to ' + output)
        thread = Downloader('https://thispersondoesnotexist.com/image', output)
        thread.start()
        time.sleep(INTERVAL)
