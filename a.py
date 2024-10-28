import socket
import json
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
from keras.src.export.export_lib import TFSMLayer

TFS_HOST = 'localhost'
TFS_PORT = 8500
MODEL_PATH = r'C:\Users\15517\PycharmProjects\MyPixcakeModel\modal\mobilenet_technical'  # 硬编码模型路径

# 全局变量，用于存储模型
model = None


def load_model():
    """加载模型"""
    return tf.keras.Sequential([
        TFSMLayer(MODEL_PATH, call_endpoint='image_quality')
    ])


def load_image(img_file, target_size):
    """加载并调整图像大小。"""
    return np.asarray(tf.keras.preprocessing.image.load_img(img_file, target_size=target_size))


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


def handle_request(conn):
    """处理来自Java的请求"""
    try:
        data = conn.recv(1024).decode('utf-8')
        if not data:
            return

        print(f"Received data: {data}")
        image_path = data.strip()  # 只需图像路径

        # Load and preprocess image
        image = load_image(image_path, target_size=(224, 224))
        image = keras.applications.mobilenet.preprocess_input(image)

        # Run the model
        prediction = model.predict(np.expand_dims(image, axis=0))

        # Access quality prediction
        quality_prediction = prediction['quality_prediction'][0]
        result = round(calc_mean_score(quality_prediction), 3)

        # 返回评分
        response = json.dumps({'mean_score_prediction': result})
        conn.sendall(response.encode('utf-8'))
    except Exception as e:
        print(f"Error handling request: {e}")
    finally:
        conn.close()


def run_server():
    global model
    model = load_model()  # 加载模型

    # 设置TCP服务器
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((TFS_HOST, 5000))  # 绑定到localhost和端口5000
        server_socket.listen()
        print("Server is listening on port 5000...")

        while True:
            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")
            handle_request(conn)


if __name__ == '__main__':
    run_server()
