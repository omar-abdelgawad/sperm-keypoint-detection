from collections.abc import Callable, Iterable, Mapping
import threading
from typing import Any


class MyThread(threading.Thread):
    def __init__(self, callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running = False
        self.callback = callback

    def run(self):
        self.running = True
        # load model
        # inference
        self.callback()
        return
        # while self.running:
        #     pass

    def stop(self):
        self.running = False


# print("start")
# t = MyThread()
# t.start()
# time.sleep(2)
# t.stop()
# t.join()
# print("stop")
