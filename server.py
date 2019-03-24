from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from gevent.pywsgi import WSGIServer
from data_loader import Language
import logging
import config
from inference import inference, load_model_param

app = Flask(__name__)
CORS(app)

CHINESE = Language(vocab_file="./couplet/vocabs")
ENCODER, DECODER = load_model_param(CHINESE, config.MODEL_DIR)


@app.route('/couplet/<in_str>')
def chat_couplet(in_str):
    if len(in_str) == 0 or len(in_str) > 17:
        output = u'您的输入太长了'
    else:
        output = inference(ENCODER, DECODER, in_str,
                           model_dir=config.MODEL_DIR, language=CHINESE)
        # output = m.infer(' '.join(in_str))
        # output = ''.join(output.split(' '))
    print('上联：%s；下联：%s' % (in_str, output))
    return jsonify({'output': output})


http_server = WSGIServer(('', 5001), app)
http_server.serve_forever()
