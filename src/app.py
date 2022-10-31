import logging

from flask import Flask, request
from flask_cors import CORS

from couplet import Couplet
from poem import Poem

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


class Handler():
    def preprocess(self, context):
        return context

    def predict(self, input):
        raise NotImplementedError

    def postprocess(self, output):
        return output

    def __call__(self, context):
        input = self.preprocess(context)
        output = self.predict(input)
        return self.postprocess(output)


class PoemHandler(Handler):
    def __init__(self) -> None:
        self.poem = Poem("checkpoints/t5-poem")
        self.poem_logger = logging.getLogger(__class__.__name__)

    def preprocess(self, context):
        author = context.get("author", None)
        title = context.get("title", None)
        self.poem_logger.info(f"author: {author}, title: {title}")
        return self.poem.preprocess(author, title)

    def predict(self, input):
        self.poem_logger.info(f"predict: {input}")
        return self.poem.predict(input)

    def postprocess(self, output):
        self.poem_logger.info(f"output: {output}")
        return super().postprocess(output)


class CoupletHandler(Handler):
    def __init__(self) -> None:
        self.couplet = Couplet("checkpoints/t5-couplet")
        self.couplet_logger = logging.getLogger(__class__.__name__)

    def preprocess(self, context):
        upper = context.get("upper", None)
        self.couplet_logger.info(f"upper: {upper}")
        return self.couplet.preprocess(upper)

    def predict(self, upper):
        self.couplet_logger.info(f"predict: {upper}")
        return self.couplet.predict(upper)

    def postprocess(self, output):
        self.couplet_logger.info(f"output: {output}")
        return super().postprocess(output)


poem_processor = PoemHandler()
couplet_processor = CoupletHandler()


@app.route('/couplet', methods=['GET', 'POST'])
def couplet_handler():
    data = request.get_json()
    logger.info(f"Couplet Receive {data}")
    return couplet_processor(data)


@app.route('/poem', methods=['GET', 'POST'])
def poem_handler():
    data = request.get_json()
    logger.info(f"Poem Receive {data}")
    return poem_processor(data)


if __name__ == "__main__":
    app.run(debug=True)
