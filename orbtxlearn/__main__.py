import queue
import random
import sys
import time
from typing import List, Tuple, Any

import click
import numpy as np
import tensorflow as tf

from . import Spy, Agent, model, config

@click.group()
def main():
    pass

@main.command()
@click.option('--host', type=str, default='localhost')
@click.option('--port', type=int, default=2600)
@click.option('--save-model', default=False, help='Save model checkpoints')
@click.option('--restore-model', default=False, help='Restore model before training')
@click.option('--save-episodes', default=False, help='Save episodes to disk')
def run(host: str, port: int, save_model: bool, restore_model: bool, save_episodes:bool) -> None:

    _q: queue.Queue = queue.Queue()
    def spy_update_callback(event_type: str, value: Any, spy: Spy) -> None:
        print(f'gameon: {spy.playing}, score: {spy.score}, dir: {spy.direction}, pps: {spy.pps:.2f}')
        _q.put((event_type, value))

    agent = Agent(config.params.image_size, config.params.image_size, 3)
    spy = Spy.make_spy(host, port, config.monitor, spy_update_callback)

    done = False
    try:
        while not done:

            print('Resetting round...')
            # Wait until we're playing
            while not spy.playing:
                time.sleep(0.05)

            agent.start_new_game()

            while spy.playing:
                im = spy.screenshot(config.params.image_size)

                keydown = agent.feed(im)

                if keydown:
                    spy.keydown()
                else:
                    spy.keyup()

                # Poll for new events
                while True:
                    try:
                        event_type, value = _q.get(False)
                    except queue.Empty:
                        break

                    if event_type == 'score':
                        agent.reward(value * config.params.reward_score)
                    elif event_type == 'playing' and value == False:
                        spy.keyup()
                        agent.gameover()
                        agent.train(spy.score, spy.pps)

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