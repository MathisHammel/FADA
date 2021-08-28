import hashlib
import random
import requests
import time
import traceback
from multiprocessing import Process, Manager

def fetch_images(hashes_seen, filesystem_lock):
    while True:
        try:
            req = requests.get('https://thispersondoesnotexist.com/image', headers={'User-Agent':'fada'})
            img = req.content
            if len(img) < 50_000:
                print(f'!!! corrupted image (size {len(img)}), skipping.')
                continue
            hsh = hashlib.sha256(img).hexdigest()
            with filesystem_lock:
                if hsh not in hashes_seen:
                    fileid = int(time.time()) * 10
                    while fileid in hashes_seen.values():
                        fileid += 1
                    hashes_seen[hsh] = fileid
                    with open(f'download/{fileid}.jpg', 'wb') as fo:
                        fo.write(img)

                    # purge all hashes older than 20s
                    for hsh in list(hashes_seen.keys()):
                        if (time.time() - 60) * 10 > hashes_seen[hsh]:
                            del hashes_seen[hsh]
                    print('Download ok, watching',len(hashes_seen),'hashes')
                
                
            time.sleep(random.random() * 0.1)  # avoid process resonance sync
        except Exception as e:
            print('!!! error while downloading image')
            traceback.print_exc()
            pass  # TODO : clean error logging


if __name__ == '__main__':
    manager = Manager()
    hashes_seen = manager.dict()
    filesystem_lock = manager.Lock()
    for i in range(3):
        p = Process(target=fetch_images, args=(hashes_seen, filesystem_lock))
        p.daemon = True  # kill children when parent is killed
        p.start()
    p.join()  # never stop parent process

