import asyncio
import uvloop
import tornado.web
import tornado.websocket
from tornado.httpserver import HTTPServer
import logging
import orjson
import os
import cv2
import numpy as np
import uuid
import redis.asyncio as redis
from redis.asyncio.client import PubSub 
from tornado.websocket import WebSocketClosedError
from concurrent.futures import ProcessPoolExecutor
from services.backgroundtasks.tasks import process_frame

uvloop.install()

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

cv2.setUseOptimized(True)

worker = ProcessPoolExecutor(max_workers=2) # Sesuaikan dengan jumlah core CPU nya!

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
RESULTS_REDIS_CHANNEL = "liveness_results"

redis_client: redis.Redis = None
redis_pubsub: PubSub = None
redis_listener_task: asyncio.Task = None

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DNN_MODEL_PROTO = os.path.join(BASE_DIR, "models/utils_detector/deploy.prototxt.txt")
DNN_MODEL_WEIGHTS = os.path.join(BASE_DIR, "models/utils_detector/res10_300x300_ssd_iter_140000.caffemodel")
DNN_CONF_THRESHOLD = 0.5
DNN_INPUT_SIZE = (300, 300)
PROGRESS_DURATION = 10

face_detector = None

try:
    if not os.path.exists(DNN_MODEL_PROTO) or not os.path.exists(DNN_MODEL_WEIGHTS):
        raise FileNotFoundError(f"Model deteksi wajah tidak ditemukan di {DNN_MODEL_PROTO} atau {DNN_MODEL_WEIGHTS}")
    face_detector = cv2.dnn.readNetFromCaffe(DNN_MODEL_PROTO, DNN_MODEL_WEIGHTS)
    logger.info(f"[INFO] Model deteksi wajah DNN berhasil dimuat.")
except Exception as e:
    logger.error(f"[ERROR] Gagal memuat model deteksi wajah DNN: {e}")
    face_detector = None

class MainHandler(tornado.websocket.WebSocketHandler):
    active_connections = {}

    def check_origin(self, origin):
        return True

    async def open(self):
        self.connection_id = str(uuid.uuid4())
        MainHandler.active_connections[self.connection_id] = self
        logger.info(f"Memulai koneksi websocket baru dengan ID: {self.connection_id}")
        await self.write_message(orjson.dumps({"status": "connected", "connection_id": self.connection_id}))
        self.current_progress = 0
        self.progress_start_time = None
        self.last_face_time = None
        self.last_celery_status = None
        self.last_celery_payload = {}

    def on_close(self):
        logger.info(f"Menutup koneksi websocket dengan ID: {getattr(self, 'connection_id', 'N/A')}")
        code = self.close_code
        reason = self.close_reason
        logger.info(f"Koneksi ditutup dengan kode: {code}, alasan: {reason}")
        if hasattr(self, 'connection_id') and self.connection_id in MainHandler.active_connections:
            del MainHandler.active_connections[self.connection_id]
            logger.info(f"Handler untuk koneksi {self.connection_id} dihapus dari active_connections.")

    def _process_frame_blocking_dnn(self, frame_bytes: bytes):
        try:
            np_array = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Gagal mendekode frame dari bytes")
                return None, "Error: Gagal mendekode frame"

            if face_detector is None:
                logger.error("Detektor wajah DNN tidak tersedia di executor.")
                return None, "Error: Deteksi wajah DNN tidak tersedia di server"

            (h, w) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(cv2.resize(frame, DNN_INPUT_SIZE),
                                         1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

            face_detector.setInput(blob)
            detections = face_detector.forward()

            faces_list = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > DNN_CONF_THRESHOLD:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w - 1, endX)
                    endY = min(h - 1, endY)

                    faces_list.append(((startX, startY, endX, endY), confidence))

            if len(faces_list) == 0:
                return None, "Status: tidak ada wajah terdeteksi"

            faces_list.sort(key=lambda item: item[1], reverse=True)
            (startX, startY, endX, endY), confidence = faces_list[0]

            padding = 10
            x1 = max(0, startX - padding)
            y1 = max(0, startY - padding)
            x2 = min(frame.shape[1] - 1, endX + padding)
            y2 = min(frame.shape[0] - 1, endY + padding)

            face_img = frame[y1:y2, x1:x2]

            if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                logger.warning(f"Hasil crop wajah kosong untuk box: ({x1},{y1},{x2},{y2})")
                return None, "Status: hasil crop wajah kosong"

            is_success, buffer = cv2.imencode('.jpg', face_img)
            if not is_success:
                logger.warning("Gagal mengenkode gambar wajah")
                return None, "Error: Gagal meng-encode potongan wajah"

            face_bytes = buffer.tobytes()
            return face_bytes, "Status: Wajah terdeteksi dan siap diproses"

        except Exception as e:
            logger.error(f"Error dalam _process_frame_blocking_dnn (executor): {e}", exc_info=True)
            return None, f"Error internal saat memproses frame: {e}"

    async def on_message(self, message):
        connection_id_for_log = getattr(self, 'connection_id', 'N/A')
        logger.info(f"Pesan diterima di on_message dari koneksi {connection_id_for_log}")
        loop = asyncio.get_event_loop()
        if not isinstance(message, bytes):
            logger.error("Pesan bukan dalam format bytes")
            await self.write_message(orjson.dumps({"error": "Pesan bukan dalam format bytes"}))
            return
        try:
            if len(message) < 4:
                logger.error("Pesan terlalu pendek untuk memuat metadata")
                await self.write_message(orjson.dumps({"error": "Pesan terlalu pendek untuk memuat metadata"}))
                return
            metadata_len = int.from_bytes(message[:4], byteorder='little')
            if len(message) < 4 + metadata_len:
                logger.error("Pesan terlalu pendek untuk memuat frame")
                await self.write_message(orjson.dumps({"error": "Pesan terlalu pendek untuk memuat frame"}))
                return
            metadata_bytes = message[4:4 + metadata_len]
            frame_bytes = message[4 + metadata_len:]
            metadata = {}
            if metadata_len > 0:
                try:
                    metadata = orjson.loads(metadata_bytes)
                except orjson.JSONDecodeError as e:
                    logger.error(f"Gagal mendekode metadata JSON: {e}")
                    await self.write_message(orjson.dumps({"error": "Gagal mendekode metadata JSON"}))
                    return
            if not frame_bytes:
                logger.warning("Frame kosong")
                await self.write_message(orjson.dumps({"error": "Frame kosong"}))
                return
            if face_detector is None:
                logger.error("Model deteksi wajah tidak tersedia")
                await self.write_message(orjson.dumps({"error": "Model deteksi wajah tidak tersedia"}))
                return

            # Panggil fungsi processing blocking (DNN detect & crop) di executor agar tidak blocking I/O
            face_bytes, status_msg = await loop.run_in_executor(
                worker,
                self._process_frame_blocking_dnn,
                frame_bytes
            )

            if face_bytes is None:
                # Jika deteksi wajah gagal, kirim pesan error langsung ke klien via WebSocket
                logger.warning(f"Preprocessing blocking selesai dengan status: {status_msg}")
                self.progress_start_time = None
                self.last_face_time = None
                self.current_progress = 0 
                await self.write_message(orjson.dumps({"error": status_msg, "progress": 0, "require_reset": True}))
                return
            logger.debug(f"Wajah terdeteksi dan di-encoded: {len(face_bytes)} bytes")

            try:
                tasks = process_frame.delay(face_bytes, self.connection_id) # Meneruskan connection_id
                logger.info(f"Task celery diterima dengan ID: {tasks.id} untuk koneksi {self.connection_id}")
                if not self.progress_start_time:
                    self.progress_start_time = loop.time()
                self.last_face_time = loop.time()
                elapsed_time = loop.time() - self.progress_start_time
                self.current_progress = min(100, int((elapsed_time / PROGRESS_DURATION) * 100))
                
                # Cek apakah progress sudah mencapai 100%
                if self.current_progress >= 100:
                    logger.info(f"Progress for connection {self.connection_id} reached 100%. Deciding final status.")
                    
                    final_status = "failed" # Default status jika progress 100 tapi belum ada hasil sukses

                    last_result_status = getattr(self, 'last_celery_status', None)
                    last_result_payload = getattr(self, 'last_celery_payload', {})
                    
                    # Contoh Kriteria Selesai Berhasil: Progress 100% DAN hasil Celery terakhir adalah "asli"
                    if last_result_status == "scanning" and last_result_payload.get("result", {}).get("result") == "asli":
                         final_status = "completed"
                         final_message = "Scanning completed: ASLI"
                         final_result_detail = last_result_payload.get("result") # Ambil detail hasil dari payload Celery

                    elif last_result_status == "failed":
                         final_status = "failed"
                         final_message = f"Scanning failed: {last_result_payload.get('error_message', 'Unknown error')}"
                         final_result_detail = None # Tidak ada detail hasil jika gagal
                         
                    else:
                         # Progress 100% tapi kriteria sukses tidak terpenuhi (misal, hasil terakhir "palsu" atau belum ada hasil "asli" yang diterima)
                         final_status = "failed"
                         final_message = "Scanning failed: Criteria not met within time limit."
                         final_result_detail = None # Atau ambil detail hasil terakhir yang ada jika relevan


                    logger.info(f"Connection {self.connection_id} final status: {final_status}")
                    final_payload_to_client = {
                         "status": final_status,
                         "connection_id": self.connection_id,
                         "progress": 100,
                         "message": final_message
                    }
                    if final_result_detail:
                         final_payload_to_client["result"] = final_result_detail

                    await self.write_message(orjson.dumps(final_payload_to_client))

                    # Tutup koneksi
                    self.close() 

                    # Reset state progress setelah proses selesai
                    self.progress_start_time = None
                    self.last_face_time = None
                    self.current_progress = 0
                    self.last_celery_status = None
                    self.last_celery_payload = {}
 
            except Exception as e:
                logger.error(f"Gagal mengirim tugas ke Celery: {e}", exc_info=True)
                error_progress_val = self.current_progress # Menggunakan self.current_progress
                await self.write_message(orjson.dumps({
                    "error": f"Gagal mengirim tugas ke Celery: {e}",
                    "progress": round(error_progress_val),
                    "require_reset": True
                }))
                # Jika gagal mengirim task, proses dianggap selesai dengan error
                self.progress_start_time = None
                self.last_face_time = None
                self.current_progress = 0
                self.last_celery_status = "failed_task_send" # Set status Celery agar diketahui di logika selesai
                self.last_celery_payload = {"error_message": f"Failed to send task: {e}"}

        except Exception as e:
            logger.error(f"!!! KRITICAL ERROR DI ON_MESSAGE untuk koneksi {connection_id_for_log}: {e}", exc_info=True)
            self.progress_start_time = None
            self.last_face_time = None
            self.current_progress = 0
            self.last_celery_status = "critical_error"
            self.last_celery_payload = {"error_message": f"Critical internal error: {e}"}

async def redis_listener(pubsub_client: PubSub):
    logger.info(f"Memulai Redis Pub/Sub listener untuk channel '{RESULTS_REDIS_CHANNEL}'")
    while True:
        try:
            message = await pubsub_client.get_message(ignore_subscribe_messages=True, timeout=1.0) 

            if message and message["type"] == "message":
                data = message["data"] 
                logger.debug(f"Menerima pesan dari Redis channel '{message['channel']}': {data}")

                try:
                    if isinstance(data, str):
                        payload = orjson.loads(data.encode('utf-8'))
                    else:
                        payload = orjson.loads(data)

                    connection_id = payload.get("connection_id")
                    task_id = payload.get("task_id")
                    status_from_celery = payload.get("status") # Ambil status dari Celery
                    if not connection_id or not task_id or status_from_celery not in ["scanning", "failed"]:
                        logger.warning(f"Pesan Redis tidak valid atau tidak lengkap (status: {status_from_celery}): {payload}")
                        continue 

                    logger.info(f"Menerima hasil task {task_id} (status: {status_from_celery}) dari Redis untuk koneksi {connection_id}")
                    handler = MainHandler.active_connections.get(connection_id)
                    logger.info(f"Handler untuk koneksi {connection_id}: {handler}")

                    if handler:
                        try:
                            # Ambil progress terakhir yang dihitung di on_message
                            current_progress = getattr(handler, 'current_progress', 0)
                            payload["progress"] = round(current_progress)
                            handler.last_celery_status = status_from_celery
                            handler.last_celery_payload = payload
                            await handler.write_message(orjson.dumps(payload)) 
                            logger.info(f"Intermediate result task {task_id} sent to client {connection_id} with status {status_from_celery}.")

                        except WebSocketClosedError:
                            logger.warning(f"Koneksi WebSocket {connection_id} tertutup saat mencoba mengirim result task {task_id}. Hasil tidak terkirim.")
                        except Exception as e:
                            logger.error(f"Error saat memproses/mengirim result task {task_id} to client {connection_id}: {e}", exc_info=True)

                    else:
                        logger.warning(f"Koneksi {connection_id} untuk task {task_id} tidak ditemukan di dictionary 'active_connections'. Hasil tidak terkirim.")

                except orjson.JSONDecodeError as e:
                    logger.error(f"Gagal mendekode pesan Redis JSON: {e}\nData: {data}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error saat memproses pesan Redis: {e}\nData: {data}", exc_info=True)

        except asyncio.CancelledError:
            logger.info("Redis listener task dibatalkan.")
            break 
        except Exception as e:
            logger.error(f"Error tak terduga di Redis listener task: {e}", exc_info=True)
            await asyncio.sleep(1.0)

async def cleanup_redis(loop):
    global redis_listener_task, redis_client
    logger.info("Memulai cleanup Redis...")
    if redis_listener_task:
        logger.info("Membatalkan Redis listener task...")
        redis_listener_task.cancel()
        try:
            await redis_listener_task
        except asyncio.CancelledError:
            logger.info("Redis listener task berhasil dibatalkan.")
        except Exception as e:
            logger.error(f"Error saat menunggu Redis listener task selesai: {e}")

    if redis_client:
        logger.info("Menutup koneksi Redis...")
        await redis_client.close()
        logger.info("Koneksi Redis ditutup.")
    await asyncio.sleep(0.1)

async def main():
    global redis_client, redis_pubsub, redis_listener_task

    if face_detector is None:
        logger.error("Server tidak dapat dimulai karena dependensi (Detektor Wajah DNN) gagal dimuat.")
        return None
    logger.info(f"Menghubungkan ke Redis di {REDIS_URL}...")
    try:
        # Koneksi asinkron ke Redis
        redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("[INFO] Koneksi Redis berhasil.")
        redis_pubsub = redis_client.pubsub()
        await redis_pubsub.subscribe(RESULTS_REDIS_CHANNEL)
        logger.info(f"[INFO] Berhasil subscribe ke Redis channel '{RESULTS_REDIS_CHANNEL}'.")
        redis_listener_task = asyncio.create_task(redis_listener(redis_pubsub))
        logger.info("[INFO] Redis listener task dimulai.")

    except Exception as e:
        logger.error(f"[ERROR] Gagal terhubung atau menginisialisasi Redis Pub/Sub: {e}")
    settings = {
        'active_connections': MainHandler.active_connections, 
    }

    app = tornado.web.Application([
        (r"/websocket", MainHandler),
    ], websocket_ping_interval=20, websocket_ping_timeout=75, **settings)

    logger.info("Memulai server WebSocket (1 proses) di port 8888")

    server = HTTPServer(app)
    server.listen(8888)
    await asyncio.Event().wait()

if __name__ == '__main__':
    import tornado.autoreload
    tornado.autoreload.start()
    logger.info("Hot reloading diaktifkan...")
    try:
        logger.info("Menjalankan server dalam mode single process (asyncio.run).")
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server dihentikan oleh pengguna.")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cleanup_redis(loop))
    except Exception as e:
        logger.error(f"Terjadi error fatal di startup: {e}", exc_info=True)
