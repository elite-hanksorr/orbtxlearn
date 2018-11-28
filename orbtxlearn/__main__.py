import queue
import random
import sys
import time
from typing import List, Tuple, Any

import click
import numpy as np
import tensorflow as tf
import pyautogui

from . import Spy, Agent, model, config

SCREENSHOT_SIZE = 480
MONITOR = 1

@click.group()
def main():
    pass

@main.command()
@click.option('--host', type=str, default='localhost')
@click.option('--port', type=int, default=2600)
def run(host: str, port: int) -> None:

    _q: queue.Queue = queue.Queue()
    def spy_update_callback(event_type: str, value: Any, spy: Spy) -> None:
        print(f'gameon: {spy.playing}, score: {spy.score}, dir: {spy.direction}, pps: {spy.pps:.2f}')
        _q.put((event_type, value))

    agent = Agent(SCREENSHOT_SIZE, SCREENSHOT_SIZE, 3)
    spy = Spy(host, port, MONITOR, spy_update_callback)

    random_episodes_left = config.training.minimum_random_episodes
    done = False
    episodes_until_eval = 5
    try:
        while not done:

            print('Resetting for {} round...'.format('collection' if episodes_until_eval else 'eval'))
            # Wait until we're playing
            while not spy.playing:
                time.sleep(0.05)

            agent.start_new_game()

            while spy.playing:
                im = spy.screenshot(SCREENSHOT_SIZE)

                if episodes_until_eval == 0:
                    keydown = agent.feed(im)
                else:
                    keydown = agent.collect(im)

                if keydown:
                    pyautogui.keyDown('space')
                else:
                    pyautogui.keyUp('space')

                # Poll for new events
                while True:
                    try:
                        event_type, value = _q.get(False)
                    except queue.Empty:
                        break

                    if event_type == 'score':
                        agent.reward(value * config.params.reward_score)
                    elif event_type == 'playing' and value == False:
                        pyautogui.keyUp('space')
                        agent.gameover()
                        if random_episodes_left > 0:
                            agent.randomize_model()
                            random_episodes_left -= 1
                        else:
                            agent.train(spy.score, spy.pps)

            episodes_until_eval -= 1
            if episodes_until_eval < 0:
                episodes_until_eval = 5
            spy.round_reset()

    finally:
        print('Closing agent')
        agent.close()


@main.command('model')
def rebuild_model():
    tf.reset_default_graph()
    with tf.summary.FileWriter(config.get_log_dir()) as writer, tf.Session() as sess:
        inputs, outputs, layers = model.make_model(1, config.params.image_size, config.params.image_size, 3)
        o_inputs, o_outputs, _ = model.make_optimizer(1, outputs['logits'])
        writer.add_graph(sess.graph)

        return

        sess.run(tf.global_variables_initializer())
        print(sess.run([outputs['logits'], outputs['action']], feed_dict={
            inputs['images']: np.zeros([1, config.params.image_size, config.params.image_size, 3])
        }))

        for i in range(100):
            sess.run(o_outputs['optimizer'], feed_dict={
                inputs['images']: np.zeros([1, config.params.image_size, config.params.image_size, 3]),
                o_inputs['actions']: np.array([0]),
                o_inputs['rewards']: np.array([-1000])
            })

            print(sess.run([outputs['logits'], outputs['action']], feed_dict={
                inputs['images']: np.zeros([1, config.params.image_size, config.params.image_size, 3])
            }))


if __name__ == '__main__':
    main()