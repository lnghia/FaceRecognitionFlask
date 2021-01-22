import os
import cv2
import json
import base64
import config
import numpy as np
from utils import generate_dataset_festures, predict_people, predict_image_as_name, delete_image_dir, delete_feature_file
from flask import Flask, request, render_template, jsonify
# from flask_socketio import SocketIO, emit

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app)

generate_dataset_festures()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit-name', methods=['POST'])
def submit_name():
    if request.method == 'POST':
        name = request.form['name'].lower()
        if name not in os.listdir(config.data_dir):
            return "SUCCESS"
        return "FAILURE"
        
@app.route('/submit-photos', methods=['POST'])
def submit_photos():
    if request.method == 'POST':
        name = request.form['name'].lower()
        images = json.loads(request.form['images'])

        os.mkdir(os.path.join(config.data_dir, str(name)))

        person_directory = os.path.join(config.data_dir, name)
        for i, image in enumerate(images):
            image_numpy = np.fromstring(base64.b64decode(image.split(",")[1]), np.uint8)
            image = cv2.imdecode(image_numpy, cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(person_directory, str(i) + '.png'), image)
        
        generate_dataset_festures()

        return "results"

@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/predict-frame", methods=['POST'])
def predict_frame():
    if request.method == 'POST':
        image = request.form['image']
        image_numpy = np.fromstring(base64.b64decode(image.split(",")[1]), np.uint8)
        image = cv2.imdecode(image_numpy, cv2.IMREAD_COLOR)

        image = predict_people(image)

        retval, buffer = cv2.imencode('.png', image)
        img_as_text = base64.b64encode(buffer)

        return img_as_text

@app.route('/api/submit-photos/', methods=['POST'])
def submit_photos_api():
    if not request.json:
        return jsonify({'is_success': 0, 'error': 'Bad request'}), 400
    if not 'left' in request.json or not 'front' in request.json or not 'right' in request.json:
        return jsonify({'is_success': 0, 'error': 'Request body must contain keys: \'left\', \'front\', \'right\''}), 400

    id = request.json['id']
    # images = json.loads(request.form['images'])
    images = [request.json['left'], request.json['front'], request.json['right']]

    os.mkdir(os.path.join(config.data_dir, str(id)))

    person_directory = os.path.join(config.data_dir, str(id))
    for i, image in enumerate(images):
        image_numpy = np.fromstring(base64.b64decode(image), np.uint8)
        image = cv2.imdecode(image_numpy, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(person_directory, str(i) + '.png'), image)
        
    generate_dataset_festures()

    return jsonify({'is_success': 1, 'error': None})

@app.route('/api/submit-id/', methods=['POST'])
def submit_id():
    if not request.json or not 'id' in request.json:
        return jsonify({'is_success': 0, 'error': 'Request body must contains \'id\''}), 400

    id = str(request.json['id'])

    if id not in os.listdir(config.data_dir):
        return jsonify({'is_success': 1, 'error': None})

    return jsonify({'is_success': 0, 'error': 'Id already exists'}), 400

@app.route('/api/identify-photo/', methods=['POST'])
def predict_photo_api():
    if not request.json or not 'image' in request.json:
        return jsonify({'is_success': 0, 'error': 'Request body must contain \'image\''}), 400

    image = request.json['image']

    image_numpy = np.fromstring(base64.b64decode(image), np.uint8)
    image = cv2.imdecode(image_numpy, cv2.IMREAD_COLOR)

    name = predict_image_as_name(image)

    # retval, buffer = cv2.imencode('.png', image)
    # img_as_text = base64.b64encode(buffer)

    return jsonify({'is_success': 1, 'data':{'id': name}, 'error': None})

@app.route('/api/forget-person/', methods=['POST'])
def forget_person():
    if not request.json or not 'id' in request.json:
        return jsonify({'is_success': 0, 'error': 'Request body must contain \'id\''}), 400

    id = str(request.json['id'])
    print(id)

    if delete_image_dir(id) and delete_feature_file():
        generate_dataset_festures()
        return jsonify({'is_success': 1, 'data': None, 'error': None})
    return jsonify({'is_success': 0, 'data': None, 'error': 'Internal server error.'}), 500




if __name__ == "__main__":
    # socketio.run(app, debug=False)
    app.run(debug=False)
