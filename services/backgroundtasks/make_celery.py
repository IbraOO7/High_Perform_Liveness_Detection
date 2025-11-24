from celery import Celery

from services.config import Config

config = Config()

REDIS_URL = config.REDIS_URL

app = Celery(
    __name__,
    broker=REDIS_URL,
    backend=REDIS_URL
)

app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['application/json', 'application/x-python-serialize'],
    enable_utc=True,
    timezone='Asia/Jakarta'
)