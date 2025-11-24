# High_Perform_Liveness_Detection

This is a snippet of a service project i've worked on. Tornado is superior and very mature in real-time connections and long polling, thanks to its low-level, native websocket with very low overhead (Linux epoll/kqueue for BSD unix), and is supported by uvloop for ultra-high concurrency and distributed background tasks, namely Celery. This project is specifically designed to handle thousands of users simultaneously for facial recognition and attendance.

How to use:
    - python3 high_liveness_test.py

run celery in other terminal:
    - celery -A services.backgroundtasks.tasks --pool=prefork (can also use threads/gevent for I/O bound) -l info

Note: this is for linux / unix env, not ideal for windows (because uvloop is not compatible in windows)

Catatan: Project ini akan stabil ketika di running di linux / unix base.

