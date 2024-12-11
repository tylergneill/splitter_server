from flask import Flask, render_template, request
from flask_restx import Api, Resource, fields
import os
import tensorflow as tf
import configuration, helper_functions, data_loader

app = Flask(__name__)
app.config["DEBUG"] = False
app.config["SECRET_KEY"] = "asdlkvumnxlapoiqyernxnfjtuzimzjdhryien"

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

@app.route("/", methods=["GET"])
def home():
    """Serve the legacy landing page."""
    return render_template("index.html")

# Swagger API setup
api = Api(
    app,
    title="2018 EMNLP Sanskrit Splitter Server API",
    description="API for inference against 2018 EMNLP Sanskrit Splitter model",
    doc="/docs",  # Swagger UI available at /docs
)

# Namespace for API endpoints
ns = api.namespace("split", description="2018 EMNLP Sanskrit Splitter API", path="/api/split")

# Input model for Swagger
split_input_model = ns.model(
    "SplitInput",
    {
        "input_text": fields.String(
            required=True,
            description="Text to be split",
            example="tava karakamalasthāṃ sphāṭikīmakṣamālāṃ , nakhakiraṇavibhinnāṃ dāḍimībījabuddhyā |",
        ),
    },
)

# Output model for Swagger
split_output_model = ns.model(
    "SplitOutput",
    {
        "output_text": fields.String(
            description="Split text result",
            example="tava kara-kamala-sthām sphāṭikīm-akṣa-mālām , nakha-kiraṇa-vibhinnām dāḍimī-bīja-buddhyā |"
        ),
    },
)

# Directories and paths
config = configuration.config
splitter_app_path = os.path.abspath(os.path.dirname(__file__))
dir_in = os.path.join(splitter_app_path, 'data/input/')
dir_out = os.path.join(splitter_app_path, 'data/output/')
path_in = os.path.join(dir_in, 'buffer_in.txt')
path_out = os.path.join(dir_out, 'buffer_out.txt')


class Splitter:
    def __init__(self):
        try:
            self.data = data_loader.DataLoader(dir_in, config, load_data_into_ram=False, load_data=False)
        except Exception as e:
            print("Error loading data: {}".format(e))

        graph_pred = tf.Graph()

        try:
            graph_pred.as_default()
        except Exception as e:
            print("Error setting graph_pred as default: {}".format(e))

        try:
            self.sess = tf.compat.v1.Session(graph=graph_pred)
        except Exception as e:
            print("Error creating TensorFlow session: {}".format(e))

        model_dir = os.path.normpath(os.path.join(splitter_app_path, config['model_directory']))
        tf.saved_model.loader.load(
            self.sess,
            [tf.saved_model.tag_constants.SERVING],
            model_dir
        )

        self.x_ph = graph_pred.get_tensor_by_name('inputs:0')
        self.split_cnts_ph = graph_pred.get_tensor_by_name('split_cnts:0')
        self.dropout_ph = graph_pred.get_tensor_by_name('dropout_keep_prob:0')
        self.seqlen_ph = graph_pred.get_tensor_by_name('seqlens:0')
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


@ns.route("/", methods=["POST"])
class SplitAPI(Resource):
    @ns.expect(split_input_model)
    @ns.marshal_with(split_output_model)
    def post(self):
        """Split Sanskrit text into meaningful units."""
        data = request.json
        input_text = data["input_text"]

        # Write input to buffer
        with open(path_in, 'w') as buffer_in:
            buffer_in.write(input_text)

        # Perform splitting
        S = Splitter()
        S.analyze()

        # Read and return the output
        with open(path_out, 'r') as buffer_out:
            output_text = buffer_out.read()

        return {"output_text": output_text}


if __name__ == "__main__":
    app.run()
