#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import bottle
from bottle import route, run
import threading

from params import Params
from process import *
from time import sleep

app = bottle.Bottle()
query = []
response = ""

@app.get("/")
def home():
    with open('demo.html', 'r') as fl:
        html = fl.read()
        return html

@app.post('/answer')
def answer():
    passage = bottle.request.json['passage']
    question = bottle.request.json['question']
    global query, response
    query = (passage, question)
    while not response:
        sleep(0.1)
    return {"answer": query}

class Demo(object):
    def __init__(self, model):
        threading.Thread(target=self.demo_backend, args = [model]).start()
        app.run(port=8080, host='0.0.0.0')

    def demo_backend(self, model):
        global query, response
        dict_ = pickle.load(open(Params.data_dir + "dictionary.pkl","r"))

        with model.graph.as_default():
            sv = tf.train.Supervisor()
            with sv.managed_session() as sess:
                sv.saver.restore(sess, tf.train.latest_checkpoint(Params.logdir))
                while True:
                    if query:
                        print(query)
                        data, shapes = dict_.realtime_process(query)
                        fd = {m:np.reshape(d, shapes[i]) for i,(m,d) in enumerate(zip(model.data[:-1], data))}
                        ids = sess.run([model.output_index], feed_dict = fc)
                        passage, question = data
                        passage_t = tokenize_corenlp(passage)
                        response = " ".join(passage_t[ids[0]:ids[1]])
