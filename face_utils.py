import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FaceRecognizerDNN:
    def __init__(self, model_path: str, proto_path: str, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        self.known_faces = {}

    def detect_faces(self, frame: np.ndarray) -> list:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    faces.append(face_crop)
        return faces

    def register_face(self, name: str, face_img: np.ndarray):
        self.known_faces[name] = face_img

    def recognize_face(self, face_img: np.ndarray) -> tuple:
        min_score = float("inf")
        matched_name = None

        face_gray = cv2.cvtColor(cv2.resize(face_img, (100, 100)), cv2.COLOR_BGR2GRAY)

        for name, known_img in self.known_faces.items():
            try:
                known_gray = cv2.cvtColor(cv2.resize(known_img, (100, 100)), cv2.COLOR_BGR2GRAY)
                score = np.mean((known_gray.astype("float") - face_gray.astype("float")) ** 2)
                if score < min_score:
                    min_score = score
                    matched_name = name
            except Exception as e:
                logger.warning(f"Failed to compare face with {name}: {e}")

        if min_score < 2000:
            return matched_name, float(min_score)
        return None, float(min_score)

    def load_known_faces_from_dir(self, dir_path: str):
        for fname in os.listdir(dir_path):
            path = os.path.join(dir_path, fname)
            if not os.path.isfile(path):
                continue
            name, _ = os.path.splitext(fname)
            img = cv2.imread(path)
            if img is not None:
                self.register_face(name, img)
                logger.info(f"[FaceRecog] Wajah '{name}' dimuat.")

_face_recognizer = None

def init_face_detector(model_path: str, proto_path: str, faces_dir: str = None):
    global _face_recognizer
    _face_recognizer = FaceRecognizerDNN(model_path, proto_path)
    logger.info("[FaceUtils] Face detector diinisialisasi.")
    if faces_dir:
        _face_recognizer.load_known_faces_from_dir(faces_dir)
        logger.info(f"[FaceUtils] Wajah dikenal dimuat dari {faces_dir}.")

def detect_face(frame: np.ndarray) -> list:
    if _face_recognizer is None:
        raise RuntimeError("Face detector belum diinisialisasi. Panggil init_face_detector terlebih dahulu.")
    return _face_recognizer.detect_faces(frame)

def recognize_face(face_img: np.ndarray) -> tuple:
    if _face_recognizer is None:
        raise RuntimeError("Face recognizer belum diinisialisasi. Panggil init_face_detector terlebih dahulu.")
    return _face_recognizer.recognize_face(face_img)
