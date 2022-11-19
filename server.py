import base64
import glob
import numpy as np
import os
import ssl
import threading
import time
import traceback

from elasticsearch import Elasticsearch
from elasticsearch.connection import create_ssl_context
from elasticsearch import helpers
from flask import Flask, abort, request, render_template
from werkzeug.utils import secure_filename
from imageio import imread
from keras.models import load_model
from PIL import Image


FACENET_PATH = 'facenet/facenet_keras.h5'
IMAGE_SIZE = 160
BATCH_SIZE = 5
THUMB_SIZE = 80
THUMB_QUALITY = 90

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def load_images(filenames, crop=True):
    images = []
    for filename in filenames:
        image = imread(filename)
        if image.shape[2] == 4:
            image = image[:,:,0:3]  # Extract only RGB
        if crop:
            image = np.array(Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS))
        images.append(image)
    return images

def align_images(images):

    aligned_images = []
    for img in images:
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            img = np.array(Image.fromarray(img).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS))
        aligned_images.append(img)

    return np.array(aligned_images)

def calc_embs_raw(images, model, batch_size=BATCH_SIZE):
    t0 = time.time()
    aligned_images = prewhiten(images)
    print(f'Image whitening time: {time.time() - t0:.02f}s')
    pred = []
    for start in range(0, len(aligned_images), batch_size):
        t0 = time.time()
        batch = aligned_images[start:start+batch_size]
        pred.append(model.predict_on_batch(batch))
        print(f'Prediction time: {time.time() - t0:.02f}s (batch size {len(batch)})')
    embs = l2_normalize(np.concatenate(pred))
    return embs

def index_vectors(elastic, vectors, ids, index='fada'):
    assert len(vectors) == len(ids)
    queries = []
    for vector, timestamp in zip(vectors, ids):
        queries.append({
            '_index':index,
            '_id':timestamp,
            'embedding':vector
        })
    #elastic.index(index, {'embedding':embs[0]}, id=ids[0])
    helpers.bulk(elastic, queries)

def indexer_thread(model, elastic):
    first_batch_done = False
    while True:
        try:
            avail_filenames = glob.glob('download/*')
            # Process oldest files first
            print(f'New cycle, {len(avail_filenames)} files (need {BATCH_SIZE} to process).')
            if len(avail_filenames) >= BATCH_SIZE or not first_batch_done:
                print('New batch, starting process.')
                if first_batch_done:
                    filenames = sorted(avail_filenames)[:BATCH_SIZE]
                else:
                    # Process a single image to force build the single-image inference function and make queries faster
                    print('Processing first batch of a single image.')
                    time.sleep(3)  # wait a bit to make sure at least 1 img in download
                    first_batch_done = True
                    filenames = [min(avail_filenames)]  # speedup first request
                images = load_images(filenames)
                print('Images loaded.')
                aligned_images = align_images(images)
                embs = calc_embs_raw(aligned_images, model, batch_size=BATCH_SIZE)
                print('Embeddings computed.')
                ids = [filename.replace('download/','').replace('.jpg','') for filename in filenames]
                index_vectors(elastic, embs, ids, 'fada')
                print('Indexing ok')
                for filename in filenames:
                    thumb = Image.open(filename).resize((THUMB_SIZE, THUMB_SIZE), Image.ANTIALIAS)
                    thumb.save(filename.replace('download/','thumbnails/'), optimize=True, quality=THUMB_QUALITY)
                    os.remove(filename)
                print('Resizing done.')
                global nb_embeddings
                nb_embeddings += len(filenames)
            else:
                time.sleep(1)
        except Exception as e:
            print('Error while processing batch.')
            traceback.print_exc()  # TODO : clean error logging
            try:
                if avail_filenames:
                    error_filename = min(avail_filenames)
                    print('Moving', error_filename, error_filename.replace('downloads/', 'errors/'))
                    os.rename(error_filename, error_filename.replace('downloads/', 'errors/'))
            except:
                print('!!! Error while moving broken file')

            time.sleep(3)


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

@app.route("/", methods=["GET"])
def index():
    global nb_embeddings
    million_embeddings = f'{nb_embeddings / 1_000_000 : .3f}'
    return render_template("index.html", million_embeddings=million_embeddings)


@app.route("/results", methods=["POST"])
def results():
    if request.files:
        image = request.files["image"]
        filename = str(int(time.time())) +  secure_filename(image.filename)
        if filename == '':
            return 'Filename is empty.'

        file_ext = os.path.splitext(filename)[1]
        if file_ext not in ['.jpg', '.png', '.jpeg', '.bmp', '.jfif']: #jfif is default ext on TPDNE
            abort(400)

        fspath = os.path.join('uploads', filename)
        image.save(fspath)
        single_image_list = load_images([fspath])
        aligned_imagelist = align_images(single_image_list)
        embedding = calc_embs_raw(aligned_imagelist, app.config['RESNET_MODEL'], batch_size=1)[0]

        search_body = {
            "size": 3,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": embedding,
                            "k": 3
                        }
                    }
                }
            }
        results = app.config['ELASTIC_INSTANCE'].search(index='fada', body=search_body)
        result_ids = [hit['_id'] for hit in results['hits']['hits']]
        result_scores = [hit['_score'] for hit in results['hits']['hits']]
        result_b64_list = []
        with open(fspath, 'rb') as im_fi:
            im_enc = base64.b64encode(im_fi.read()).decode('ascii')
        for result_id in result_ids:
            with open(os.path.join('thumbnails', result_id + '.jpg'), 'rb') as thumb_fi:
                encoded = base64.b64encode(thumb_fi.read()).decode('ascii')
                result_b64_list.append((encoded, result_id))
        resstr = '<html><body>\n' # TODO: make this a template (it started as a prototype but now it's a clusterfuck plz anyone make a PR)
        resstr += '<h1>Top 3 matches:</h1><br/>\n'
        for enc in result_b64_list:
            resstr += '<img src="data:image/jpeg;base64,'+enc[0]+'" width="200"><img src="data:image/jpeg;base64,'+im_enc+'" width="200"><br/>\n'
            resstr += '<p>Image timestamp: ' + str(int(enc[1])//10) + '</p><br/>\n' # TODO: convert to date
        resstr += '<br/><br/>\n'
        im = Image.open(fspath)
        mask = Image.open('mask.png').resize(im.size)
        im.paste(mask, None, mask)
        eyespath = os.path.join('eyes',filename+'.jpg')
        im.save(eyespath)
        with open(eyespath, 'rb') as eyes_fi:
            eyes_enc = base64.b64encode(eyes_fi.read()).decode('ascii')
        resstr += '<h1>Eye mask technique:</h1><br/>\n'
        resstr += '<img src="data:image/jpeg;base64,'+eyes_enc+'" width="400">\n'
        resstr += '</body></html>'
        return resstr

if __name__ == '__main__':
    print('Starting up...')

    # Uncomment this to force CPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    """
    # Setup GPU with enough RAM
    device = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device[0], True)
    tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
    print('GPU OK.')
    """
    model = load_model(FACENET_PATH)
    model.compile()
    model.make_predict_function()
    print('Model loaded and compiled.')

    open_distro_ssl_context = create_ssl_context()
    open_distro_ssl_context.check_hostname = False
    open_distro_ssl_context.verify_mode = ssl.CERT_NONE
    elastic = Elasticsearch(
        scheme="https",
        hosts=[ { 'port': 9200, 'host': 'localhost' } ],
        ssl_context=open_distro_ssl_context,
        http_auth=("fada_elastic", os.getenv('FADA_ELASTIC_PASSWORD')),
        timeout=30,
        verify_certs=True
    )

    elastic.indices.refresh('fada')
    global nb_embeddings
    nb_embeddings = int(elastic.cat.count('fada', params={"format": "json"})[0]['count'])

    print(f'Elasticsearch connected, {nb_embeddings} embeddings in the index.')

    idx_thread = threading.Thread(target=indexer_thread, args=(model, elastic))
    idx_thread.start()
    
    app.config['ELASTIC_INSTANCE'] = elastic
    app.config['RESNET_MODEL'] = model
    app.run('0.0.0.0', port=443, threaded=True, debug=False, ssl_context=('/etc/letsencrypt/live/fada.h25.io/cert.pem',
                                                                           '/etc/letsencrypt/live/fada.h25.io/privkey.pem'))
