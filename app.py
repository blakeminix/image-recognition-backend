from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/upload', methods=['POST'])
def upload_image():
    # Handle image upload logic here (e.g., save to S3)
    # Example: Access uploaded file using request.files['file']
    print("Received image!!")
    return jsonify({'message': 'Image uploaded successfully'})

if __name__ == '__main__':
    app.run(debug=True)