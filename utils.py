import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
import cv2

# Initialize detector and FaceNet model
detector = MTCNN()
embedder = FaceNet()

def extract_face(image, required_size=(160, 160)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(image)

    if len(results) == 0:
        return None

    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)

    return face

def get_embedding(face):
    face = face.astype('float32')
    face = np.expand_dims(face, axis=0)
    embedding = embedder.embeddings(face)
    return embedding[0]
