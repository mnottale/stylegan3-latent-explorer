#! /usr/bin/env python3


import os
import math
import sys
import base64
import json
import threading
import io

import numpy as np

import torch
from torchvision import transforms, utils

from PIL import Image

latent_dim = 512

class Channel():
    def __init__(self):
        self.l_queue = threading.Lock()
        self.s_not_empty = threading.Condition(self.l_queue)
        self.queue = list()
    def push(self, item):
        with self.l_queue:
            self.queue.append(item)
            if len(self.queue) == 1:
                self.s_not_empty.notify()
    def pop(self):
        with self.l_queue:
            while len(self.queue) == 0:
                self.s_not_empty.wait()
            item = self.queue[0]
            self.queue = self.queue[1:]
        return item
    def current_queue_length(self):
        with self.l_queue:
            return len(self.queue)

class TaskSerializer():
    def __init__(self, processor, initializer=None):
        self.channel = Channel()
        self.processor = processor
        self.initializer = initializer
    def process(self, tsk):
        l = threading.Semaphore(0)
        result = list()
        self.channel.push((tsk, l, result))
        l.acquire()
        return result[0]
    def start_in_thread(self):
        self.thread = threading.Thread(target=self.wrap_run)
        self.thread.start()
    def wrap_run(self):
        try:
            self.run()
        except Exception as e:
            print('BRONK: {} {}'.format(e, e.backtrace()))
    def run(self):
        if self.initializer is not None:
            self.initializer()
        while True:
            payload = self.channel.pop()
            res = self.processor(payload[0])
            payload[2].append(res)
            payload[1].release()
    def current_queue_length(self):
        return self.channel.current_queue_length()

# GAN stuff
device = torch.device('cuda')
G = None
network_pkl = 'stylegan3.pkl'
sys.path.append("stylegan3")
import dnnlib
import legacy

def stylegan_init():
    global G
    print("opening network...")
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    print("...done,")
    

def stylegan_go(l):
    img = G(torch.Tensor(np.array([l])).to(device), None, truncation_psi = 1, noise_mode = 'const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img[0].cpu().numpy()
    return img

# latent encoding stuff
def v_to_bin(v):
    res = b''
    for f in v:
        if f < -2.99:
            f = -2.99
        elif f > 2.99:
            f = 2.99
        ival = int((f + 3.0) / 6.0 * 16777216.0)
        res += bytes([ival%256, (ival//256)%256, (ival//65536)%256])
    return res

def v_to_b64(v):
    return base64.b64encode(v_to_bin(v)).decode()

def v_from_bin(b):
    if len(b) % 3 != 0:
        raise Exception("bad data size")
    res = list()
    for i in range(len(b)//3):
        v1 = int(b[3*i+0])
        v2 = int(b[3*i+1])
        v3 = int(b[3*i+2])
        vv = v1 + (v2 * 256) + (v3 * 65536)
        f = vv / 16777216.0 * 6.0 - 3.0
        res.append(f)
    return res

def v_from_b64(b):
    return v_from_bin(base64.b64decode(b.encode()))


def latent_randomize(count):
    if count > 100:
        raise Exception("too many vector requested")
    v = torch.randn([count, latent_dim]).tolist()
    v = list(map(lambda x: v_to_b64(x), v))
    v = json.dumps(v)
    return v

def latent_around(ecenter, count, length):
    if count > 100:
        raise Exception("too many vector requested")
    center = torch.Tensor(v_from_b64(ecenter))
    res = list()
    for i in range(count):
        lat = center + torch.randn([latent_dim])*length
        res.append(v_to_b64(lat.tolist()))
    return json.dumps(res)



def application(environ, start_response):
    url = environ['PATH_INFO']
    if url == '/' or url == '/index.html':
        with open('stylegan-client.html', 'rb') as f:
            payload = f.read()
        status = '200 OK'
        headers = [('Content-type', 'text/html')]
        start_response(status, headers)
        return [payload]
    comps = url.split('/')[1:]
    if comps[1] == 'load':
        l = stylegan_processor.current_queue_length()
        status = '200 OK'
        headers = [('Content-type', 'application/json')]
        start_response(status, headers)
        return [str(l).encode()]
    if comps[1] == 'image':
        q = environ['QUERY_STRING']
        p=q.index('=')
        center = q[p+1:]
        latent = v_from_b64(center)
        image = stylegan_processor.process(latent)
        img_byte_arr = io.BytesIO()
        Image.fromarray(image).save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        status = '200 OK'
        headers = [('Content-type', 'image/jpeg')]
        start_response(status, headers)
        return [img_byte_arr]
    if comps[1] == 'random':
        count = int(comps[2])
        j = latent_randomize(count)
        status = '200 OK'
        headers = [('Content-type', 'application/json')]
        start_response(status, headers)
        return [j.encode()]
    if comps[1] == 'around':
        count = int(comps[2])
        radius = float(comps[3])
        q = environ['QUERY_STRING']
        p=q.index('=')
        center = q[p+1:]
        j = latent_around(center, count, radius)
        status = '200 OK'
        headers = [('Content-type', 'application/json')]
        start_response(status, headers)
        return [j.encode()]
    status = '404 Not Found'
    headers = [('Content-type', 'text/plain')]
    start_response(status, headers)
    return ['Nope'.encode()]



stylegan_processor = TaskSerializer(stylegan_go, stylegan_init)
# DEBUG MOGE WITHOUT STYLEGAN
#def dummy_image(l):
#    return np.array(Image.open('/tmp/image.jpg'))
#
#stylegan_processor = TaskSerializer(dummy_image)
stylegan_processor.start_in_thread()

from waitress import serve

serve(application, host='0.0.0.0', port=8001)
