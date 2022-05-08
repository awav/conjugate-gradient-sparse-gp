from typing import Callable, Optional, Union, Dict
from pathlib import Path
from tensorboardX import SummaryWriter
import numpy as np


class Monitor:
    __initial_iteration: int = 0

    def __init__(self, logdir: Union[Path, str]):
        max_queue = 5
        flush_secs = 10
        self._logdir = logdir
        self._iter: int = self.__initial_iteration
        self._writer = SummaryWriter(logdir=str(logdir), max_queue=max_queue, flush_secs=flush_secs)
        self._callbacks = {}

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    def add_callback(self, name: str, callback: Callable, record_step: Optional[int] = None):
        self._callbacks[name] = (callback, record_step, {})

    def reset(self):
        self._iter = self.__initial_iteration
        self.writer.flush()
        callbacks = {}
        for name, (cb, record_step, _) in self._callbacks.items():
            callbacks[name] = (cb, record_step, {})
        self._callbacks = callbacks

    def flush(self):
        self.writer.flush()
        logdir = self._logdir
        for k, (_, _, v) in self._callbacks.items():
            path = Path(logdir, f"{k}.logs.npy")
            store_logs(path, v)

    def close(self):
        self.flush()
        self.writer.close()

    def _increment_iter(self):
        self._iter += 1

    def _handle_callback(self, step: int, name: str):
        cb, record_step, logs = self._callbacks[name]

        if record_step is not None and step % record_step != 0:
            return

        results = cb(step)
        for key, res in results.items():
            if isinstance(res, (list, np.ndarray)) and _len(res) > 1:
                for i, r in enumerate(res):
                    self.writer.add_scalar(f"{key}_{i}", r, global_step=step)
            else:
                self.writer.add_scalar(key, res, global_step=step)

            if key in logs:
                logs[key].append(res)
            else:
                logs[key] = [res]

    def __call__(self, step: int, *args, **kwargs):
        internal_step = self._iter
        for name in self._callbacks:
            self._handle_callback(internal_step, name)
        self._increment_iter()


def _len(obj) -> int:
    if isinstance(obj, np.ndarray):
        return obj.size
    return len(obj)


def store_logs(path: Path, logs: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, logs, allow_pickle=True)
