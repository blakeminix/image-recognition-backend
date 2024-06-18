from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
import boto3
import uuid
import threading
import json
import requests

app = Flask(__name__)
CORS(app)

BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, '.env.local'))

S3_BUCKET = os.getenv('S3_BUCKET')
ACCESS_KEY_ID = os.getenv('ACCESS_KEY_ID')
SECRET_ACCESS_KEY = os.getenv('SECRET_KEY_ID')
KUBERNETES_SERVICE_URL = os.getenv('KUBERNETES_SERVICE_URL')

s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY,
)

def initiate_processing(filename):
    try:
        response = requests.post(
            f'{KUBERNETES_SERVICE_URL}/process_image',
            json={'filename': filename}
        )

        if response.status_code == 200:
            print('Processing initiated successfully.')
        else:
            print('Failed to initiate processing.')

    except Exception as e:
        print(f"Error initiating processing: {e}")


@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    filename = str(uuid.uuid4()) + '-' + file.filename
    print('Filename:', filename)

    try:
        s3_client.upload_fileobj(file, S3_BUCKET, filename)
        print('File uploaded to S3.')
        threading.Thread(target=initiate_processing, args=(filename,)).start()
        return jsonify({'message': 'File uploaded and processing initiated', 'filename': filename}), 200
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return jsonify({'message': 'Error uploading file to S3'}), 500


@app.route('/api/retrieve/<filename>', methods=['GET'])
def retrieve_prediction(filename):
    try:
        result_filename = f'{filename}.json'
        result_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=result_filename)
        result_data = json.loads(result_obj['Body'].read().decode('utf-8'))
        return jsonify(result_data), 200
    except Exception as e:
        print(f"Error retrieving prediction: {e}")
        return jsonify({'message': 'Prediction not ready'}), 202
    finally:
        try:
            s3_client.delete_object(Bucket=S3_BUCKET, Key=result_filename)
            print(f"Deleted result from S3: {filename}")
        except Exception as e:
            print(f"Error deleting result from S3: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
