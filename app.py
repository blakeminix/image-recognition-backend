from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
import boto3
import uuid
import threading
import json
import requests
import tensorflow as tf
from PIL import Image

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

MODEL_FILE_KEY = 'image_recognition_model_cifar100.h5'
MODEL_LOCAL_PATH = 'image_recognition_model_cifar100.h5'

def download_model():
    if not os.path.exists(MODEL_LOCAL_PATH):
        try:
            s3_client.download_file(S3_BUCKET, MODEL_FILE_KEY, MODEL_LOCAL_PATH)
            print(f'Model downloaded from S3 to {MODEL_LOCAL_PATH}')
        except Exception as e:
            print(f'Error downloading model: {e}')

download_model()

# TF model
model = tf.keras.models.load_model('image_recognition_model_cifar100.h5')

# Model's labels
labels = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

def initiate_processing(filename):
    try:
        print(f'Initiating processing for {filename}')
        response = requests.post(
            f'{KUBERNETES_SERVICE_URL}/process_image',
            json={'filename': filename},
        )
        print(f'Response received for {filename}')
        if response.status_code == 200:
            print('Processing initiated successfully.')
        else:
            print('Failed to initiate processing.')

    except Exception as e:
        print(f"Error initiating processing: {e}")


def process_image(filename):
    try:
        local_filename = filename
        s3_client.download_file(S3_BUCKET, filename, local_filename)
        print('File downloaded locally.')

        img = Image.open(local_filename)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img = img.resize((32, 32))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array /= 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_index = tf.argmax(prediction[0]).numpy()
        predicted_label = labels[predicted_index]

        result = {
            'prediction': prediction.tolist(),
            'predicted_label': predicted_label
        }

        result_filename = f'{filename}.json'
        s3_client.put_object(Bucket=S3_BUCKET, Key=result_filename, Body=json.dumps(result))
        print(f"Prediction stored in S3: {result_filename}")

    except Exception as e:
        print(f"Error processing image: {e}")
        error_result = {'error': 'Error processing image'}
        result_filename = f'{filename}.json'
        s3_client.put_object(Bucket=S3_BUCKET, Key=result_filename, Body=json.dumps(error_result))

    finally:
        try:
            os.remove(local_filename)
            print(f"Deleted local file: {local_filename}")
        except Exception as e:
            print(f"Error deleting file: {e}")

        try:
            s3_client.delete_object(Bucket=S3_BUCKET, Key=filename)
            print(f"Deleted file from S3: {filename}")
        except Exception as e:
            print(f"Error deleting file from S3: {e}")


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
        # threading.Thread(target=initiate_processing, args=(filename,)).start()
        threading.Thread(target=process_image, args=(filename,)).start()
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