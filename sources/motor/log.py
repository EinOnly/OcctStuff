import datetime
import threading
import queue
import sys
from loguru import logger as COLORLOG

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

COLORLOG.configure(
    handlers=[
        {"sink": sys.stdout, "level": "DEBUG", "format": "<level>{message}</level>"},
        #  {"sink": "./Logcache/DemoLog{time:YYYY-MM}.log",
        #   "level": "DEBUG",
        #   "rotation": "10 MB",
        #   "format": "<level>{message}</level>"}
    ]
)


# LOG#############
class Log:
    class logdata:
        def __init__(self, data: str, tag: int) -> None:
            self.m_Data = data
            self.m_Tag = tag

    def __init__(self) -> None:
        self.m_DataQueue = queue.Queue()
        self.lock = threading.Lock()
        self.m_Thread = threading.Thread(target=self.Update)
        self.m_Thread.start()
        self.m_Stop = False

    def sync(self):
        while not self.m_DataQueue.empty():
            continue
        self.m_Stop = True
        self.m_Thread.join()

    def Update(self):
        while not self.m_Stop:
            if not self.m_DataQueue.empty():
                with self.lock:
                    data = self.m_DataQueue.get()
                    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    if data.m_Tag == 0:  # info
                        COLORLOG.info(f"{GREEN}[{time}][INFO] {data.m_Data}{RESET}")

                    if data.m_Tag == 1:  # warningng
                        COLORLOG.warning(f"{YELLOW}[{time}][warn] {data.m_Data}{RESET}")

                    if data.m_Tag == 2:  # error
                        COLORLOG.error(f"{RED}[{time}][ERRO] {data.m_Data}{RESET}")
            else:
                continue

    def info(self, str):
        with self.lock:
            logData = self.logdata(str, 0)
            self.m_DataQueue.put(logData)

    def warn(self, str):
        with self.lock:
            logData = self.logdata(str, 1)
            self.m_DataQueue.put(logData)

    def erro(self, str):
        with self.lock:
            logData = self.logdata(str, 2)
            self.m_DataQueue.put(logData)


########LOG########
CORELOG = Log()
########LOG########

if __name__ == "__main__":
    CORELOG.info("info")
    CORELOG.warn("warn")
    CORELOG.erro("error")
    CORELOG.sync()
