from flask import Flask, redirect, render_template, request, url_for, session, send_from_directory, send_file, make_response
app = Flask(__name__)
app.config["DEBUG"] = True
app.config["SECRET_KEY"] = "asdlkvumnxlapoiqyernxnfjtuzimzjdhryien" # for session, no actual need for secrecy

import sys,os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import configuration, helper_functions, data_loader

config = configuration.config

path_in = os.path.join(os.getcwd(), 'data/input', 'buffer_in.txt')
path_out = os.path.join(os.getcwd(), 'data/output', 'buffer_out.txt')

class Splitter:

    def __init__(self):

        try:
            self.data = data_loader.DataLoader('data/input', config, load_data_into_ram=False, load_data = False)
        except:
            print("couldn't load data")

        graph_pred = tf.Graph()

        try:
            graph_pred.as_default()
        except:
            print("couldn't do graph_pred.as_default()")

        try:
            self.sess = tf.Session(graph=graph_pred)
        except:
            print("couldn't get tensorflow session")

        model_dir = model_dir = os.path.normpath( os.path.join(os.getcwd(), config['model_directory']) )
        tf.saved_model.loader.load(
            self.sess,
            [tf.saved_model.tag_constants.SERVING],
            model_dir
        )

        self.x_ph = graph_pred.get_tensor_by_name('inputs:0')
        self.split_cnts_ph = graph_pred.get_tensor_by_name('split_cnts:0')
        self.dropout_ph = graph_pred.get_tensor_by_name('dropout_keep_prob:0')
        self.seqlen_ph  = graph_pred.get_tensor_by_name('seqlens:0')
        self.predictions_ph = graph_pred.get_tensor_by_name('predictions:0')

    def analyze(self):

        helper_functions.analyze_text(
            path_in,
            path_out,
            self.predictions_ph,
            self.x_ph,
            self.split_cnts_ph,
            self.seqlen_ph,
            self.dropout_ph,
            self.data,
            self.sess,
            verbose=True
        )

S = Splitter()

@app.route("/", methods=["POST"])
def index():
    if request.method == "GET":
        return "POST requests only please (and don't forget to use input in valid IAST)"
    elif request.method == "POST":
        data = request.get_json()
        try:
            input_text = data["input_text"]
        except:
            print("couldn't find input_text parameter")
        with open(path_in, 'w') as buffer_in:
            buffer_in.write(input_text)
        S.analyze()
        with open(path_out, 'r') as buffer_out:
            return buffer_out.read()