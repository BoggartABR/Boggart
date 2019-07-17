from SimpleHTTPServer import SimpleHTTPRequestHandler
from BaseHTTPServer import HTTPServer
import json
import numpy as np
import tensorflow as tf
from constants import *
from abr_algs.pensieve import Pensieve

PORT = 9996
NN_MODEL = 'abr_algs/pensieve/models/lin_model.ckpt'


def make_request_handler(state, pensieve):

    class Handler(SimpleHTTPRequestHandler):

        state = np.zeros((INPUT_LEN, HISTORY_LEN))

        def __init__(self,request, client_address, server):
            self.pensieve = pensieve
            SimpleHTTPRequestHandler.__init__(self, request, client_address, server)


        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(content_length))

            last_quality = data["last_quality"]
            bitrates = data["bitrates"]
            buffer_size = data["buffer_size"]
            throughput = data["throughput"]
            delay = data["delay"]
            next_sizes = data["next_sizes"]
            video_chunk_remain = data["video_chunk_remain"]
            total_chunks = data["total_chunks"]

            Handler.state = np.roll(Handler.state, -1, axis=1)

            # this should be S_INFO number of terms
            Handler.state[LAST_BR_IDX, -1] = last_quality
            Handler.state[BUFFER_IDX, -1] = buffer_size
            Handler.state[THROUGHPUT_IDX, -1] = throughput
            Handler.state[DELAY_IDX, -1] = float(delay)
            Handler.state[NEXT_CHUNKS_START_IDX, :self.video[BR_DIM]] = np.array(next_sizes)  # mega byte
            Handler.state[CHUNKS_TILL_END_IDX, -1] = np.minimum(video_chunk_remain, total_chunks)

            bit_rate = self.pensieve.get_quality(Handler.state)

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', len(bit_rate))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(bit_rate)

    return Handler


def run_server(video, reward):

    with tf.Session() as sess:
        pensieve = Pensieve.Pensieve(video, reward, NN_MODEL)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters
        # restore neural net parameters
        saver.restore(sess, NN_MODEL)
        state = np.zeros((INPUT_LEN, HISTORY_LEN))
        handler = make_request_handler(state, pensieve)
        httpd = HTTPServer(("", PORT), handler)

        print "serving at port", PORT
        httpd.serve_forever()
