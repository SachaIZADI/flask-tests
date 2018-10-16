from flask import Flask, request, jsonify, send_file
import os
import torch
from PIL import Image
import io
import numpy as np


import sys
sys.path.insert(0, "static/models")



app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if request.files.get("image"):

            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = np.asarray(image)
            image = image / 255

            dirname = os.path.dirname(__file__)
            file_name = os.path.join(dirname, 'static', 'models', 'conv_class.pk')
            mnist_clf = torch.load(file_name)

            y_pred = mnist_clf(torch.from_numpy(image).resize_(1, 1, 28, 28).float())
            y_pred = torch.max(y_pred.data, 1)
            proba = np.exp(y_pred[0].numpy()[0])
            label_pred = y_pred[1].numpy()[0]

            if y_pred == 0:
                y_pred = 8

            return jsonify({"predicted_label":int(label_pred), "probability":float(proba)})



@app.route('/return_image', methods=['POST'])
def return_image():
    if request.method == 'POST':
        if request.files.get("image"):

            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = np.asarray(image)
            image = image / 255

            I = Image.fromarray(image * 255)
            I = I.convert("L")
            imgByteArr = io.BytesIO()
            I.save(imgByteArr, format='JPEG')
            imgByteArr.seek(0)

            return send_file(
                imgByteArr,
                mimetype='image/jpeg',
                as_attachment=True,
                attachment_filename='result.jpg')


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=5000)


# curl -X POST -F image=@eight.jpg 'http://localhost:5000/return_image' -o img.jpg