# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import pickle
import numpy as np
import pandas as pd
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType


input_sample = pd.DataFrame(data=[{'cycle': 1.0, 'setting1': 0.0023, 'setting2': 0.0003, 'setting3': 100.0, 's1': 518.67, 's2': 643.02, 's3': 1585.29, 's4': 1398.21, 's5': 14.62, 's6': 21.61, 's7': 553.9, 's8': 2388.04, 's9': 9050.17, 's10': 1.3, 's11': 47.2, 's12': 521.72, 's13': 2388.03, 's14': 8125.55, 's15': 8.4052, 's16': 0.03, 's17': 392.0, 's18': 2388.0, 's19': 100.0, 's20': 38.86, 's21': 23.3735}], columns=['cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'])
output_sample = np.array([0])


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = Model.get_model_path(model_name = 'AutoML916c88b0f44')
    model = joblib.load(model_path)


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
