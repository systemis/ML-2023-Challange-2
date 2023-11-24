from flask import Flask, render_template, jsonify
from sklearn.preprocessing import MinMaxScaler

import math
import tensorflow as tf
import numpy as np

import yfinance as yf

app = Flask(__name__)

yf.pdr_override()
df = yf.download('BTC-USD', start='2014-01-01', end='2022-01-10', threads=False)     # or threads=True
dataset = df.filter(['Close']).values
training_data_len = math.ceil(len(dataset) * .8)

# Load model from file model.keras
model = tf.keras.models.load_model("model2.h5")

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
    valid = data[training_data_len:]
    valid['Predictions'] = predictions_2
    
    # Replace this with your actual data
    predictions_data = {
      'date': df.index.tolist()[training_data_len:],
      'close': valid['Close'].tolist(),
      'predictions': valid['Predictions'].tolist(),
    }
    return jsonify(predictions_data)

if __name__ == '__main__':
    app.run(debug=True, port=8080)