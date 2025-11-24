import os
import time
import orjson
import redis
import numpy as np
import cv2
import logging
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from celery.signals import worker_process_init
import threading
from celery.utils.log import get_task_logger
from services.config import Config
from .make_celery import app

config = Config()

REDIS_URL = os.environ.get(config.REDIS_URL, "redis://localhost:6379/0")

redis_publisher_pool = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)
redis_publisher = redis.Redis(connection_pool=redis_publisher_pool)

# Nama channel Redis untuk hasil
RESULTS_REDIS_CHANNEL = "liveness_results"

IMG_SIZE = (224, 224)
SPOOF_THRESHOLD = 0.6
CLASS_NAMES = ['real', 'spoof']
SAVE_FRAME = True
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "antispoofing_models/antispoofing_model_20250426_000007.keras")

logger = get_task_logger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

model_load_lock = threading.Lock()

antispoofing_model = None

# Konfigurasi GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        logger.info("[INFO] Memori GPU diatur untuk pertumbuhan dinamis.")
    except RuntimeError as e:
        logger.error(f"[ERROR] Gagal mengatur memori GPU: {e}")
else:
    logger.warning("[WARNING] Tidak ada GPU yang terdeteksi, menggunakan CPU.")

def load_and_warmup_model():
    global antispoofing_model
    try:
        logger.info("[Model Loader] Memuat model antispoofing...")
        loaded_model = tf.keras.models.load_model(MODEL_DIR, compile=False)
        logger.info("[Model Loader] Model antispoofing berhasil dimuat.")

        logger.info("[Model Loader] Melakukan warmup model...")
        dummy_input = np.random.rand(1, *IMG_SIZE, 3).astype(np.float32)
        dummy_input = preprocess_input(dummy_input)
        _ = loaded_model.predict(dummy_input, verbose=0)
        logger.info("[Model Loader] Warmup model selesai.")
        antispoofing_model = loaded_model
    except Exception as e:
        logger.error(f"[Model Loader] Gagal memuat atau melakukan warmup model: {e}", exc_info=True)
        antispoofing_model = None

@worker_process_init.connect
def on_worker_process_init(**kwargs):
    logger.info("Sinyal worker_process_init diterima. Memulai pemuatan model...")
    load_and_warmup_model()
    if antispoofing_model is not None:
        logger.info("Model berhasil dimuat dan di-warmup saat worker process init.")
    else:
        logger.error("GAGAL memuat model saat worker process init. Worker mungkin tidak berfungsi dengan benar.")

def get_model():
    if antispoofing_model is None:
        logger.warning("[get_model] Model belum dimuat! Mungkin ada masalah saat startup worker. Mencoba memuat sekarang...")
        with model_load_lock:
            if antispoofing_model is None:
                load_and_warmup_model()
    return antispoofing_model

def decode_image_bytes(frame_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        logger.error("Gagal decode frame dari bytes. Ukuran bytes diterima: %d", len(frame_bytes))
        raise ValueError("Gagal decode frame dari bytes.")
    return frame

def preprocess_image(frame: np.ndarray) -> np.ndarray:
    try:
        if len(frame.shape) != 3 or frame.shape[2] != 3:
             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        resized = cv2.resize(frame, IMG_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        preprocessed = np.expand_dims(rgb, axis=0).astype(np.float32)
        return preprocess_input(preprocessed)
    except Exception as e:
         logger.error(f"Error dalam preprocess_image: {e}", exc_info=True)
         raise ValueError(f"Gagal preprocess gambar: {e}")


def predict_liveness(frame_input: np.ndarray) -> dict:
    model = get_model()
    if model is None:
        logger.error("[Predict] Model tidak tersedia di worker ini.")
        return {
            "result": "error_model_unavailable",
            "confidence": 0.0,
            "message": "Model AI tidak termuat di worker"
        }

    try:
        predict = model.predict(frame_input, verbose=0)
        if 'spoof' not in CLASS_NAMES or 'real' not in CLASS_NAMES:
             logger.error(f"CLASS_NAMES tidak mengandung 'spoof' atau 'real': {CLASS_NAMES}")
             raise ValueError("CLASS_NAMES tidak valid")

        try:
            spoof_prob = float(predict[0][CLASS_NAMES.index('spoof')])
        except (IndexError, TypeError) as e:
            logger.error(f"Gagal mendapatkan probabilitas spoof dari output prediksi: {predict}. Error: {e}")
            raise ValueError("Gagal parse output prediksi model")

        label_predict = 'asli' if spoof_prob < SPOOF_THRESHOLD else 'palsu'

        logger.info(f"[Predict] Hasil: {label_predict}, Prob: {spoof_prob:.4f}")

        return {
            "result": label_predict,
            "confidence": spoof_prob,
            "message": "Prediksi berhasil"
        }
    except Exception as e:
        logger.error(f"[Predict] Error saat menjalankan prediksi: {e}", exc_info=True)
        return {
            "result": "error_prediction_failed",
            "confidence": 0.0,
            "message": f"Error saat prediksi: {e}"
        }

def process_pipeline(frame_bytes: bytes):
    try:
        frame = decode_image_bytes(frame_bytes)
        input_tensor = preprocess_image(frame)
        result = predict_liveness(input_tensor)
        result['nama'] = "NN" # Definisikan nama orang berdasarkan foto
    except ValueError as ve:
        logger.error(f"ValueError dalam process_pipeline: {ve}", exc_info=True)
        return {'result': 'error_pipeline_value', 'confidence': 0.0, 'message': f'Pipeline processing value error: {str(ve)}'}
    except Exception as e:
        logger.error(f"Error umum dalam process_pipeline: {e}", exc_info=True)
        return {'result': 'error_pipeline_generic', 'confidence': 0.0, 'message': f'Pipeline processing generic error: {str(e)}'}


@app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_frame(self, frame_bytes: bytes, connection_id: str):
    logger.info(f"[Task {self.request.id}] Menerima frame size: {len(frame_bytes)} untuk koneksi: {connection_id}")

    callback_payload = {
        "task_id": str(self.request.id),
        "connection_id": connection_id,
    }

    try:
        prediction_output = process_pipeline(frame_bytes) # Simpan hasil ke variabel baru
        if isinstance(prediction_output, dict) and prediction_output.get("result", "").startswith("error_"):
            logger.error(f"[Task {self.request.id}] Pipeline gagal: {prediction_output.get('message', 'Tidak ada pesan error')}")
            callback_payload["status"] = "failed"
            callback_payload["result"] = prediction_output # Mengirim detail error dari pipeline
            callback_payload["error_message"] = prediction_output.get('message', 'Unknown pipeline error')
        elif not isinstance(prediction_output, dict): # Jika bukan dictionary, ada masalah serius
            logger.error(f"[Task {self.request.id}] Pipeline tidak mengembalikan dictionary yang diharapkan. Diterima: {type(prediction_output)}")
            callback_payload["status"] = "failed"
            callback_payload["result"] = None
            callback_payload["error_message"] = "Pipeline returned unexpected data type. Expected a dictionary."
        else:
            # Jika berhasil dan merupakan dictionary
            logger.info(f"[Task {self.request.id}] Task selesai. Hasil: {prediction_output.get('result')}")
            callback_payload["status"] = "scanning"
            callback_payload["result"] = prediction_output
    except Exception as e:
        logger.error(f"[Task {self.request.id}] Gagal memproses frame karena exception di luar pipeline: {e}", exc_info=True)
        callback_payload["status"] = "failed"
        callback_payload["result"] = None # Tidak ada hasil prediksi
        callback_payload["error_message"] = f"Task execution error: {str(e)}"
    
    try:
        message_data = orjson.dumps(callback_payload)
        redis_publisher.publish(RESULTS_REDIS_CHANNEL, message_data)
        logger.info(f"[Task {self.request.id}] Hasil task dipublikasikan ke channel '{RESULTS_REDIS_CHANNEL}'. Payload size: {len(message_data)}")
    except Exception as e:
        logger.error(f"[Task {self.request.id}] Gagal mem-publish hasil ke Redis: {e}", exc_info=True)
    
    return callback_payload