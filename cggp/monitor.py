from typing import Callable, Union, Dict
from pathlib import Path
from tensorboardX import SummaryWriter
import numpy as np


class Monitor:
    def __init__(self, logdir: Union[Path, str], record_step: Union[int, None] = None):
        max_queue = 5
        flush_secs = 10
        self._logdir = logdir
        self._iter: int = 0
        self._record_step: Union[None, int] = record_step
        self._writer = SummaryWriter(logdir=str(logdir), max_queue=max_queue, flush_secs=flush_secs)
        self._callbacks = {}

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    def add_callback(self, name: str, callback: Callable):
        self._callbacks[name] = (callback, {})

    def reset(self):
        self._iter = 0
        self.writer.flush()
        callbacks = {}
        for name, (cb, _) in self._callbacks.items():
            callbacks[name] = (cb, {})
        self._callbacks = callbacks

    def flush(self):
        self.writer.flush()
        logdir = self._logdir
        for k, (_, v) in self._callbacks.items():
            path = Path(logdir, f"{k}.logs.npy")
            store_logs(path, v)

    def close(self):
        self.flush()
        self.writer.close()

    def _increment_iter(self):
        self._iter += 1

    def _handle_callback(self, step: int, name: str):
        cb, logs = self._callbacks[name]
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
