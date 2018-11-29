from abc import abstractmethod, ABC
import socket
import sys
import threading
import time
from typing import Optional, Callable, Any, Tuple, Union

import mss
import numpy as np
import skimage.transform

if sys.platform.startswith('linux'):
    import xdo
elif sys.platform == 'win32':
    import ctypes
else:
    raise RuntimeError(f'Not implemented for platform {sys.platform} yet')

__all__ = ['Spy']

class Spy(ABC):
    DIR_CW = -1
    DIR_CCW = 1
    DIR_UNINITIALIZED = 0

    def __init__(self, host: str = 'localhost', port: int = 2600, monitor: int = 1,
                 callback: Optional[Callable[[str, Any, 'Spy'], None]] = None):
        self._host = host
        self._port = port
        self._callback = callback

        self._playing = False
        self._score = 0
        self._direction = Spy.DIR_UNINITIALIZED

        self._sct = mss.mss()
        self._sct_monitor = self._sct.monitors[monitor]

        self._time_start = time.time()
        self._time_end: Optional[float] = None

        self._send_lock = threading.Lock()

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind((host, port))
        self._socket.listen()
        self._conn = self._socket.accept()[0]

        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

    @abstractmethod
    def keydown(self, key: str) -> None:
        pass

    @abstractmethod
    def keyup(self, key: str) -> None:
        pass

    def _send(self, data: str) -> None:
        with self._send_lock:
            self._conn.send((data + '\n').encode('utf8'))

    def _do_callback(self, event_type: str, value: Any) -> None:
        if self._callback is not None:
            self._callback(event_type, value, self)

    def _reset_timer(self) -> None:
        self._time_end = None
        self._time_start = time.time()

    def _end_timer(self) -> None:
        self._time_end = time.time()

    def _read_loop(self) -> None:
        conn_file = self._conn.makefile('r', encoding='utf8')
        for line in conn_file:
            line = line.strip().lower()

            if line == 'event:gameon':
                self._playing = True
                self._reset_timer()
                self._do_callback('playing', True)
            elif line == 'event:gameoff':
                self._playing = False
                self._end_timer()
                self._do_callback('playing', False)
            elif line == 'dir:cw':
                self._direction = Spy.DIR_CW
                self._do_callback('direction', self._direction)
            elif line == 'dir:ccw':
                self._direction = Spy.DIR_CCW
                self._do_callback('direction', self._direction)
            elif line.startswith('score:'):
                old_score = self._score
                self._score = int(line.partition(':')[2])
                self._do_callback('score', self._score - old_score)

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def score(self) -> int:
        return self._score

    @property
    def direction(self) -> int:
        return self._direction

    @property
    def elapsed(self) -> float:
        if self._time_end is not None:
            return self._time_end - self._time_start
        else:
            return time.time() - self._time_start

    @property
    def pps(self) -> float:
        elapsed = self.elapsed
        if elapsed <= 0:
            return 0
        return self.score / elapsed

    def round_reset(self) -> None:
        while self._time_end is not None and time.time() - self._time_end <= 3:
            time.sleep(0.05)

        self._send('command:restart')

    def screenshot(self, size: int) -> np.ndarray:
        '''Takes a square screenshot'''
        # start = time.time()

        im = np.array(self._sct.grab(self._sct_monitor))
        # print(im, im.shape, time.time() - start)
        h, w, channels = im.shape
        assert channels >= 3

        # Make square
        smaller = min(w, h)
        im = im[(h-smaller)//2:(h+smaller)//2, (w-smaller)//2:(w+smaller)//2, 0:3]
        assert im.shape[0] == im.shape[1]
        # print(im, im.shape, time.time() - start)

        # Resize
        im = skimage.img_as_ubyte(skimage.transform.resize(im, [size, size], mode='reflect', anti_aliasing=False))
        # print(im, time.time() - start)

        im[:,:,[0,2]] = im[:,:,[2,0]]
        # print(im, time.time() - start)
        # import pdb; pdb.set_trace()
        return im
