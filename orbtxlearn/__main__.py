import random
import sys
import time

import pyautogui

from . import Spy

def spy_update_callback(event_type: str, spy: Spy) -> None:
    print(f'gameon: {spy.playing}, score: {spy.score}, dir: {spy.direction}, pps: {spy.pps:.2f}')
    if event_type == 'state' and not spy.playing:
        time.sleep(3)
        spy.round_reset()

def main(host: str = 'localhost', port: int = 2600) -> None:
    spy = Spy(host, port, spy_update_callback)
    while True:
        if spy.playing:
            time.sleep(random.random())
            pyautogui.keyDown('space')
            time.sleep(random.random())
            pyautogui.keyUp('space')
        else:
            time.sleep(0.05)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        main(sys.argv[1], int(sys.argv[2]))
    else:
        main()