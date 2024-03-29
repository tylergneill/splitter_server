from flask import Flask, render_template, request
app = Flask(__name__)
app.config["DEBUG"] = True
app.config["SECRET_KEY"] = "asdlkvumnxlapoiqyernxnfjtuzimzjdhryien" # for session, no actual need for secrecy

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import configuration, helper_functions, data_loader

config = configuration.config

splitter_app_path = os.path.abspath(os.path.dirname(__file__))
dir_in = os.path.join(splitter_app_path, 'data/input/')
dir_out = os.path.join(splitter_app_path, 'data/output/')
path_in = os.path.join(dir_in, 'buffer_in.txt')
path_out = os.path.join(dir_out, 'buffer_out.txt')

class Splitter:

    def __init__(self):

        try:
            self.data = data_loader.DataLoader(dir_in, config, load_data_into_ram=False, load_data = False)
        except:
            print("couldn't load data")

        graph_pred = tf.Graph()

        try:
            graph_pred.as_default()
        except:
            print("couldn't do graph_pred.as_default()")

        try:
            self.sess = tf.compat.v1.Session(graph=graph_pred)
        except:
            print("couldn't get tensorflow session")

        model_dir = os.path.normpath( os.path.join(splitter_app_path, config['model_directory']) )
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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        if request.json:
            data = request.get_json()
            try:
                input_text = data["input_text"]
            except:
                print("couldn't get input_text from json data['input_text']")
        elif request.files:
            try:
                input_file = request.files["input_file"]
                input_fn = input_file.filename
                input_text = input_file.stream.read().decode('utf-8')
            except:
                print("couldn't get input_text from request.files['input_file']")

        with open(path_in, 'w') as buffer_in:
            buffer_in.write(input_text)
        S = Splitter()
        S.analyze()
        with open(path_out, 'r') as buffer_out:
            return buffer_out.read()
