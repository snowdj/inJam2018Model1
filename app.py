import numpy as np
from flask import Flask, abort, jsonify, request
import pickle
import json

myPickle = 'model.pkl'

encoder, modelCatBoost = pickle.load(open(myPickle, 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
	return "Xor Prediction!"

@app.route('/model1', methods=['POST', 'GET'])
def make_predict():
	
	## all kinds of error checking should go here
	data = request.get_json(force=True)
	## convert our json to a numpy array 
	predict_request = [data['CnaeSession'], 
                       data['Uf'], 
                       data['FundationDate'], 
                       data['Equity']]
	predict_request = np.array(predict_request)
	predict_reshape = np.array(predict_request).reshape(1, -1)

	predIndex = modelCatBoost.predict(predict_reshape).astype('int').flatten()
	predGroup = encoder.inverse_transform(predIndex)
	## print(predIndex, predGroup)
	myHardCodePred = 'G1000000'
	return json.dumps({'hello': myHardCodePred})
	## return our prediction
	##return jsonify(predIndex = predIndex.tolist(), predGroup = predGroup.tolist())
	# return jsonify(yep = "Hello world!")
	#return jsonify(myGuess = myHardCodePred)

if __name__ == '__main__':
    # app.run(port = 9000, debug = True, use_reloader=True)
	app.run(debug = True)