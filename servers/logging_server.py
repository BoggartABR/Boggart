from SimpleHTTPServer import SimpleHTTPRequestHandler
from BaseHTTPServer import HTTPServer
import json

PORT = 9997
LOG_DIR = "internet_results/"
ALGS = ['Boggart', 'Pensieve', 'Bola', 'MPC', 'Throughput', 'Buffer']

def make_request_handler(video_dict):

    class Handler(SimpleHTTPRequestHandler):

        counter = 0

        def __init__(self,request, client_address, server):
            self.video_dict = video_dict
            SimpleHTTPRequestHandler.__init__(self, request, client_address, server)


        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(content_length))
            self.send_response(200)

            with open(LOG_DIR + data['alg'], 'a') as f:
                json.dump(data['data'], f)
                f.write('\n')

            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()


    return Handler

def run_server():
    video_dict = {}
    for a in ALGS:
        video_dict[a] = {}
    handler = make_request_handler(video_dict)
    httpd = HTTPServer(("", PORT), handler)

    print "serving at port", PORT
    httpd.serve_forever()
