from SimpleHTTPServer import SimpleHTTPRequestHandler
from BaseHTTPServer import HTTPServer
PORT = 9998


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):

        SimpleHTTPRequestHandler.do_GET(self)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()


def run_server():
    httpd = HTTPServer(("", PORT), Handler)

    print "serving at port", PORT
    httpd.serve_forever()
