from flask import Flask, request, render_template, jsonify
from model import predict_file  # Import model function

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    file_path = "static/" + file.filename
    file.save(file_path)

    result = predict_file(file_path)  # Call model prediction
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
