# Serve model as a flask application

import pickle
from flask import Flask, request
import json
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model

input_sample = pd.DataFrame(data=[{'cycle': 31.0, 'setting1': 0.0034, 'setting2': -0.0002, 'setting3': 100.0, 's1': 518.67, 's2': 642.64, 's3': 1589.99, 's4': 1402.34, 's5': 14.62, 's6': 21.61, 's7': 553.46, 's8': 2388.1, 's9': 9053.49, 's10': 1.3, 's11': 47.5, 's12': 521.7, 's13': 2388.12, 's14': 8129.68, 's15': 8.4407, 's16': 0.03, 's17': 393.0, 's18': 2388.0, 's19': 100.0, 's20': 38.77, 's21': 23.4134, 'RUL': 164.0}], columns=['cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'])
print(input_sample)

model = None
app = Flask(__name__)

def load_model():
    print("load_model method invoked")
    global model
    # model variable refers to the global variable
    # with open('model_AutoML916c88b0f44.pkl', 'rb') as f:
    #     model = pickle.load(f)
    model_path = Model.get_model_path(model_name = 'model_AutoML916c88b0f44.pkl')
    model = joblib.load(model_path)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    print("predict method invoked")
    try:
        if request.method == 'POST':
            data = request.get_json()  # Get data posted as a json
            data_df = pd.DataFrame(data)
            data_df[['cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']] = pd.DataFrame(data_df.data.values.tolist(),index=data_df.index)
            data_df = data_df.drop(['data'], axis=1)
            result = model.predict(data_df)  # runs globally loaded model on the data
            return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        print(result, "\n")
        return json.dumps({"error": result})


if __name__ == '__main__':
    # load model at the beginning once only
    print("main method invoked")
    load_model()
    app.run(host='0.0.0.0', port=5000)