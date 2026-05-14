from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    # Trả về trang giao diện chính (cần tạo file index.html)
    return "Chào mừng bạn đến với Hệ thống Chẩn đoán Bệnh Thận qua ảnh CT!"

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    # Cho phép kích hoạt lại pipeline huấn luyện qua web (tùy chọn)
    os.system("dvc repro")
    return "Đã hoàn thành huấn luyện thành công!"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)

if __name__ == "__main__":
    clApp = ClientApp()
    # Chạy trên port 8080 để dễ dàng mapping với Docker
    app.run(host='0.0.0.0', port=8080)