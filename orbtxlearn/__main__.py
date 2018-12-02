import os
import queue
import random
import sys
import time
from typing import List, Tuple, Any, Optional

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
def main() -> None:
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.episodes_dir, exist_ok=True)

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

    agent = Agent('run', config.params.image_size, config.params.image_size, 3)
    if model and restore_model:
        agent.restore()

    print('Ready!')

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
                        agent.reward('score', value)
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
@click.option('--max-time', required=False, default=None, type=float, help='Maximum time (in minutes) to train')
@click.argument('epochs', type=int)
def train(restore_model: bool, max_time: Optional[float], epochs: int) -> None:
    agent = Agent('train', config.params.image_size, config.params.image_size, 3)

    if restore_model:
        agent.restore()

    if max_time is not None:
        max_time *= 60

    agent.train(epochs, max_time=max_time)


@main.command('model')
def rebuild_model() -> None:
    tf.reset_default_graph()
    with tf.summary.FileWriter(config.get_run_log_dir('test')) as writer, tf.Session() as sess:
        inputs, outputs, layers = model.make_model(config.params.image_size, config.params.image_size, 3)
        o_inputs, o_outputs, _ = model.make_optimizer(outputs['logits'])
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        print('Testing model...')
        for i in range(5):
            run_metadata = tf.RunMetadata()
            sess.run([outputs['logits'], outputs['action']], feed_dict={
                inputs['images']: np.random.uniform(0, 256, size=[1, config.params.image_size, config.params.image_size, 3]),
            }, options=run_options, run_metadata=run_metadata)

            writer.add_run_metadata(run_metadata, f'test_eval_{i}')
            writer.flush()

        print('Testing optimizer...')
        b = config.training.batch_size
        for i in range(5):
            run_metadata = tf.RunMetadata()
            sess.run(o_outputs['optimizer'], feed_dict={
                inputs['images']: np.random.uniform(0, 256, size=[b, config.params.image_size, config.params.image_size, 3]),
                o_inputs['actions']: np.random.uniform(0, 2, size=[b]).astype(np.uint8),
                o_inputs['rewards']: np.random.normal(size=[b]),
            }, options=run_options, run_metadata=run_metadata)

            writer.add_run_metadata(run_metadata, f'test_optim_{i}')
            writer.flush()


if __name__ == '__main__':
    main()