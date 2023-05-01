from http.server import SimpleHTTPRequestHandler
import socketserver
from urllib.parse import urlparse, parse_qs
import gan_handler
import sys
import time

class RequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        global index
        
        query_components = parse_qs(urlparse(self.path).query)
        #print(query_components)
        if("code" in query_components.keys()):
            # found marker but false id or no marker
            code = query_components["code"][0]
            rot = query_components['rotation'][0]
            with open(log_file, "a") as f:
                f.write(f"{index},{code},{rot}\n")
        
            rot, id, img_file = gan_handler.dump_image(index, baseline=baseline)
            if not (rot == -1 and id == -1):
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(bytes(f"{index},{id},{rot},{img_file}","utf-8")) 
                index += 1
            # end of data file
            else:
                self.send_response(201)
                print(f"processed {index} images in {time.time() - start_time} sec")
                sys.exit()

            return

        return SimpleHTTPRequestHandler.do_GET(self)

host = "localhost"
port = 8080
baseline = True
log_file = "log_artk_small_bs.csv"

# reset log file
with open(log_file, "w") as f:
    f.write(f"Image,Code,Rotation\n")

# load gan and data
gan_handler = gan_handler.GanHandler(data_file="./Data/data_pickled_artk_small.obj", gan_file="8_2_GAN_500_gpu10chi_4_gen.h5")
index = 0
start_time = time.time()

handler = RequestHandler
server = socketserver.TCPServer((host, port), handler)
print(f"Serving {host}:{port}")
server.serve_forever()