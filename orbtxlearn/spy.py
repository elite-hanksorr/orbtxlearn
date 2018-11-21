import socket
import threading
import time
from typing import Optional, Callable

__all__ = ['Spy']

class Spy():
    DIR_CW = -1
    DIR_CCW = 1
    DIR_UNINITIALIZED = 0

    def __init__(self, host: str = 'localhost', port: int = 2600,
                 callback: Optional[Callable[[str, 'Spy'], None]] = None):
        self._host: str = host
        self._port: int = port
        self._callback: Optional[Callable[[str, Spy], None]] = callback

        self._playing: bool = False
        self._score: int = 0
        self._direction: int = Spy.DIR_UNINITIALIZED

        self._time_start: float = time.time()
        self._time_end: Optional[float] = None

        self._send_lock = threading.Lock()

        self._socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind((host, port))
        self._socket.listen()
        self._conn: socket.socket = self._socket.accept()[0]

        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()

    def _send(self, data: str):
        with self._send_lock:
            self._conn.send((data + '\n').encode('utf8'))

    def _do_callback(self, *args, **kwargs):
        if self._callback is not None:
            self._callback(*args, **kwargs)

    def _reset_timer(self):
        self._time_end = None
        self._time_start = time.time()

    def _end_timer(self):
        self._time_end = time.time()

    def _read_loop(self):
        conn_file = self._conn.makefile('r', encoding='utf8')
        for line in conn_file:
            line = line.strip().lower()

            if line == 'event:gameon':
                self._playing = True
                self._reset_timer()
                self._do_callback('state', self)
            elif line == 'event:gameoff':
                self._playing = False
                self._end_timer()
                self._do_callback('state', self)
            elif line == 'dir:cw':
                self._direction = Spy.DIR_CW
                self._do_callback('direction', self)
            elif line == 'dir:ccw':
                self._direction = Spy.DIR_CCW
                self._do_callback('direction', self)
            elif line.startswith('score:'):
                self._score = int(line.partition(':')[2])
                self._do_callback('score', self)

    @property
    def playing(self):
        return self._playing

    @property
    def score(self):
        return self._score
    
    @property
    def direction(self):
        return self._direction

    @property
    def elapsed(self):
        if self._time_end is not None:
            return self._time_end - self._time_start
        else:
            return time.time() - self._time_start

    @property
    def pps(self):
        elapsed = self.elapsed
        if elapsed <= 0:
            return 0
        return self.score / elapsed

    def round_reset(self):
        self._send('command:restart')