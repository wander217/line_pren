import logging
from os.path import join, isdir
from os import mkdir
import time
import json
from typing import Dict, List


class PRENLogger:
    def __init__(self, workspace: str, level: str):
        workspace = join(workspace, 'logger')
        if not isdir(workspace):
            mkdir(workspace)
        workspace = join(workspace, 'recognizer')
        if not isdir(workspace):
            mkdir(workspace)

        self.level: int = logging.INFO if level == "INFO" else logging.DEBUG
        formater = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s')
        self.logger = logging.getLogger("message")
        self.logger.setLevel(self.level)

        file_handler = logging.FileHandler(join(workspace, "ouput.log"))
        file_handler.setFormatter(formater)
        file_handler.setLevel(self.level)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formater)
        stream_handler.setLevel(self.level)
        self.logger.addHandler(stream_handler)

        self.time: float = time.time()
        self.metric_path: str = join(workspace, "metric.txt")

    def report_time(self, name: str):
        current: float = time.time()
        self._write(name + " - time: {}".format(current - self.time))
        self.time = current

    def report_metric(self, metric: Dict):
        self.report_delimiter()
        keys: List = list(metric.keys())
        for key in keys:
            self._write("\t- {}: {}".format(key, metric[key]))
        self.write_metric(metric)
        self.report_delimiter()
        self.report_newline()

    def write_metric(self, metric: Dict):
        with open(self.metric_path, 'a', encoding='utf=8') as f:
            f.write(json.dumps(metric))
            f.write("\n")

    def report_delimiter(self):
        self._write("-" * 33)

    def report_newline(self):
        self._write("")

    def _write(self, message: str):
        if self.level == logging.INFO:
            self.logger.info(message)
            return
        self.logger.debug(message)
