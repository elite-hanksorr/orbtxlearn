import sys
import time

from . import Spy

def spy_update_callback(event_type: str, spy: Spy) -> None:
    if event_type == 'state' and not spy.playing:
        time.sleep(1.5)
        spy.round_reset()

    print(f'gameon: {spy.playing}, score: {spy.score}, dir: {spy.direction}, pps: {spy.pps:.2f}')

def main(host: str = 'localhost', port: int = 2600) -> None:
    spy = Spy(host, port, spy_update_callback)
    time.sleep(100000)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], int(sys.argv[2]))
    else:
        main()