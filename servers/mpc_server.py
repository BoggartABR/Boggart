from SimpleHTTPServer import SimpleHTTPRequestHandler
from BaseHTTPServer import HTTPServer
import json
import numpy as np
from abr_algs import mpc as mpc_object
from constants import *

PORT = 9995


def make_request_handler(mpc):

    class Handler(SimpleHTTPRequestHandler):

        state = np.zeros((INPUT_LEN, HISTORY_LEN))

        def __init__(self, request, client_address, server, video):
            self.mpc = mpc
            self.video = video
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

            bit_rate = str(self.mpc.get_quality(Handler.state))

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', len(bit_rate))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(bit_rate)

    return Handler


def run_server(video, reward):

    mpc = mpc_object.MPC(video, reward)
    handler = make_request_handler(mpc)
    httpd = HTTPServer(("", PORT), handler)

    print "serving at port", PORT
    httpd.serve_forever()
