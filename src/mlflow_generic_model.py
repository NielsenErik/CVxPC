import mlflow
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import datetime
import uuid
import json
import base64

# wrapper used for sklearn transformers to apply transformations to columns
class ModelOut (mlflow.pyfunc.PythonModel):
    def __init__(self, model, output_columns):
        self.model = model
        self.output_columns = output_columns

    def fit(self, model_input: pd.DataFrame):
        model_input = model_input.to_numpy()
        if len(model_input.shape)==1:
            model_input = model_input.reshape(-1,1)
        self.model.fit(model_input)

    def predict(self, model_input: pd.DataFrame):
        # apply transformation and obtain a dataframe with correct columns
        model_input = model_input.to_numpy()
        if len(model_input.shape)==1:
            model_input = model_input.reshape(-1,1)
        transformed_data = self.model.transform(model_input)
        
        return pd.DataFrame(transformed_data, columns=self.output_columns)
    
    def inverse_predict(self, model_input):
        if len(model_input.shape)==1:
            model_input = model_input.reshape(-1,1)
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.to_numpy()
        transformed_data = self.model.inverse_transform(model_input)
        return pd.DataFrame(transformed_data, columns=self.output_columns)
    
class PytorchWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_class, data_scaler, output_columns):
        self.model = model_class
        self.data_scaler = data_scaler
        self.output_columns = output_columns


    def load_context(self, context):
        self.model.load_state_dict(torch.load(context.artifacts["rul_model_path"]))
        self.model.eval()

    def set_data_loader(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        dataset = TensorDataset(x)
        data_loader = DataLoader(dataset, batch_size=self.model.batch_size, shuffle=False)
        return data_loader

    def eval(self, data):
        data_loader = self.set_data_loader(data)
        outputs = []
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                data = data[0].to("cpu")
                output = self.model(data)
                outputs.append(output)
        outputs = torch.cat(outputs, dim=0).detach().numpy()
        return outputs

    def predict(self, model_input: pd.DataFrame):
        model_input = self.data_scaler.predict(model_input)
        output = self.eval(np.array([model_input.to_numpy()]))
        return pd.DataFrame(output, columns=self.output_columns).astype(float)

class AlarmGenerator(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.model = None

    def predict (self, model_input: pd.DataFrame):
        alarm_envelopes = []

        for _, row in model_input.iterrows():
            
            dm_envelope = {
                "ReceivedTopic": "",
                "CorrelationID": "",
                "ApiVersion": "v2",
                "RequestID": "",
                "ContentType": "application/json",
                "ErrorCode": 0,
                "Payload": "",
                "QueryParams": {}
            }

            presentDate = datetime.datetime.now()
            unix_timestamp = datetime.datetime.timestamp(presentDate) * 1000
            rul = row['RUL']
            if isinstance(rul, np.float64):
                rul = np.float32(rul)
            message = f'Current Remaining Useful Life {rul}'
            if row['RUL']<0.10:
                severity = "Critical"
            elif row['RUL']>0.10 and row['RUL']<0.30:
                severity = "Major"
            elif row['RUL']>0.30 and row['RUL']<0.50:
                severity = "Medium"
            else:
                severity = "Normal"
            alarm_data = {
                'message': message,
                'severity': severity,  # TBD
                'resolved': False,
                'description': "RUL in OBD2 data from vehicle",
                'job': "NA",
                'annotations': None
            }
            alarm_wrapper = {
                'alarm': alarm_data,
                'status': 'Online',
                'origin': int(unix_timestamp),
                'device_name': 'Volkswagen',  # hard coded
                'id': str(uuid.uuid4())
            }
            dm_envelope["Payload"] = base64.b64encode(json.dumps(alarm_wrapper).encode('utf-8'))

            alarm_envelopes.append(dm_envelope)

        if alarm_envelopes:
            alarm_df = pd.DataFrame(alarm_envelopes)
        else:
            alarm_df = pd.DataFrame(columns=['ReceivedTopic', 'CorrelationID', 'ApiVersion', 'RequestID', 'ContentType', 'ErrorCode','Payload','QueryParams'])

        return alarm_df
