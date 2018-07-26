import numpy as np
from flask import Flask, abort, jsonify, request
import pickle
import json
from sklearn.externals import joblib


app = Flask(__name__)
@app.route('/')
def homepage():
    welcomeLabel = """
    <h1>Final Count Down!</h1>
    """
    return welcomeLabel


@app.route('/model1', methods=['POST'])
def make_predict():
    myPickle = 'model.pkl'
    encoder, modelCatBoost = pickle.load(open(myPickle))
    # encoder, modelCatBoost = pickle.load(open(myPickle, 'rb'))
    
    ##encoder, modelCatBoost = joblib.load('model.pkl')
    ## all kinds of error checking should go here
    ## convert our json to a numpy array  
    data = request.get_json(force=True)	
    predict_request = [data['CnaeSession'], 
                       data['Uf'], 
                       data['FundationDate'], 
                       data['Equity']]
    predict_request = np.array(predict_request)
    predict_reshape = np.array(predict_request).reshape(1, -1)

    # predIndex = modelCatBoost.predict(predict_reshape).astype('int').flatten()
    # predGroup = encoder.inverse_transform(predIndex)
    ## print(predIndex, predGroup)
    myHardCodePred = 'G124'
    return json.dumps({'hello': myHardCodePred})
    ## return our prediction
    ##return jsonify(predIndex = predIndex.tolist(), predGroup = predGroup.tolist())
    # return jsonify(yep = "Hello world!")
    #return jsonify(myGuess = myHardCodePred)

if __name__ == '__main__':
    app.run(debug = True)