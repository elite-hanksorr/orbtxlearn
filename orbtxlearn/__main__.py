import os
import queue
import random
import sys
import time
from typing import List, Tuple, Any

import click
import numpy as np
import tensorflow as tf

try:
    import ipdb
except ImportError:
    import pdb
else:
    pdb = ipdb

from . import Spy, Agent, model, config

@click.group()
def main():
    os.makedirs(config._log_dir, exist_ok=True)
    os.makedirs(config.episodes_dir, exist_ok=True)
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

@main.command('run')
@click.option('--host', type=str, default='localhost')
@click.option('--port', type=int, default=2600)
@click.option('--model/--no-model', default=False, help='Wether to use the model, or guess randomly')
@click.option('--restore-model/--no-restore-model', default=False, help='Restore model before running')
def run(host: str, port: int, model: bool, restore_model: bool) -> None:
    if restore_model and not model:
        raise click.UsageError('Cannot specify --restore-model without --model')

    _q: queue.Queue = queue.Queue()
    def spy_update_callback(event_type: str, value: Any, spy: Spy) -> None:
        print(f'gameon: {spy.playing}, score: {spy.score}, dir: {spy.direction}, pps: {spy.pps:.2f}')
        _q.put((event_type, value))

    agent = Agent(config.params.image_size, config.params.image_size, 3)
    if model and restore_model:
        agent.restore()

    spy = Spy.make_spy(host, port, config.monitor, spy_update_callback)

    done = False
    try:
        while not done:

            print('Resetting round...')
            # Wait until we're playing
            while not spy.playing:
                time.sleep(0.05)

            agent.start_new_game()
            keydown_prob = random.random()*0.4 + 0.3

            while spy.playing:
                im = spy.screenshot(config.params.image_size)

                if model:
                    keydown = agent.feed(im)
                else:
                    keydown = agent.collect(im, keydown_prob=keydown_prob)
                    print()

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
                        agent.report_game(spy.score, spy.pps)

            spy.round_reset()

    except Exception:
        _, _, tb = sys.exc_info()
        # pdb.post_mortem(tb)
        raise

    finally:
        print('Closing agent')
        agent.close()


@main.command('train')
@click.option('--restore-model/--no-restore-model', default=False, help='Restore model before training')
@click.argument('epochs', type=int)
def train(restore_model: bool, epochs: int) -> None:
    agent = Agent(config.params.image_size, config.params.image_size, 3)

    if restore_model:
        agent.restore()

    agent.train(epochs)


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