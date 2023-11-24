from flask import Flask, render_template, jsonify
from sklearn.preprocessing import MinMaxScaler

import math
import tensorflow as tf
import numpy as np

app = Flask(__name__)

import yfinance as yf
yf.pdr_override()

df = yf.download('BTC-USD', start='2014-01-01', end='2022-01-10', threads=False)     # or threads=True

dataset = df.filter(['Close']).values
training_data_len = math.ceil(len(dataset) * .8) # We are using %80 of the data for training


model = tf.keras.models.load_model("model2.h5")
# model.make_predict_function()

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
test_data = scaled_data[training_data_len - 60 : , :]
X_test = []
y_test = dataset[training_data_len : , :]
for i in range(60, len(test_data)):
  X_test.append(test_data[i-60 : i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    predictions_2 = model.predict(X_test)
    predictions_2 = scaler.inverse_transform(predictions_2)
    
    # Sample data (replace this with your actual data)
    data = df.filter(['Close'])
    valid_2 = data[training_data_len:]
    valid_2['Predictions'] = predictions_2
    
    # Replace this with your actual data
    predictions_data = {
        # 'valid_1': {
        #     'close': valid_1['Close'].tolist(),
        #     'predictions': valid_1['Predictions'].tolist()
        # },
        'valid_2': {
            'close': valid_2['Close'].tolist(),
            'predictions': valid_2['Predictions'].tolist()
        }
    }
    return jsonify(predictions_data)

if __name__ == '__main__':
    app.run(debug=True)