from abc import abstractmethod, ABC
import socket
import sys
import threading
import time
from typing import Optional, Callable, Any, Tuple, Union, List

import mss
import numpy as np
from PIL import Image
import pyautogui
import skimage.transform

__all__ = ['Spy']

class Spy(ABC):
    DIR_CW = -1
    DIR_CCW = 1
    DIR_UNINITIALIZED = 0

    @classmethod
    def make_spy(cls, *args, **kwargs) -> 'Spy':
        if sys.platform.startswith('linux'):
            from . import linux
            return linux.LinuxSpy(*args, **kwargs)
        elif sys.platform == 'win32':
            from . import win32
            return win32.WindowsSpy(*args, **kwargs)
        else:
            raise RuntimeError(f'Not implemented for platform {sys.platform} yet')

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

        self._screenshot_latencies: List[Tuple[float, float]] = []

    def keydown(self) -> None:
        pyautogui.keyDown('space')

    def keyup(self) -> None:
        pyautogui.keyUp('space')

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

    def _resize_ndarray(self, size: int, im: np.ndarray) -> np.ndarray:
        h, w, channels = im.shape
        assert channels >= 3

        # Make square
        smaller = min(w, h)
        im = im[(h-smaller)//2:(h+smaller)//2, (w-smaller)//2:(w+smaller)//2, 0:3]
        assert im.shape[0] == im.shape[1]

        # Resize
        im = skimage.img_as_ubyte(skimage.transform.resize(im, [size, size], mode='reflect', anti_aliasing=False))

        # BGR -> RGB
        im[:,:,[0,2]] = im[:,:,[2,0]]

        return im

    def _resize_pil(self, size, im):
        w, h = im.size

        # Make square
        smaller = min(w, h)
        im = im.crop(((w-smaller)//2, (h-smaller)//2, (w+smaller)//2, (h+smaller)//2))
        assert im.size[0] == im.size[1]

        # Resize
        im = im.resize((size, size))

        return np.array(im)

    def screenshot(self, size: int) -> np.ndarray:
        '''Takes a square screenshot'''
        time_start = time.time()
        im = np.array(self._sct.grab(self._sct_monitor))
        time_captured = time.time()
        # resized = self._resize_pil(size, np.array(im)[:,:,[2,1,0]])
        resized = self._resize_pil(size, Image.fromarray(np.array(im)[:,:,[2,1,0]], 'RGB'))
        time_resized = time.time()

        t0 = time_captured - time_start
        t1 = time_resized - time_captured
        print(f'capture: {1000*t0:6.1f}ms resize: {1000*t1:6.1f}ms', end=' ')
        self._screenshot_latencies.append((t0, t1))

        return resized
