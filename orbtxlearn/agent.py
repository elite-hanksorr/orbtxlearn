import random
import time
from typing import List, Union, Dict, Any, MutableSequence

from . import model, config

import numpy as np
import pandas as pd
import tensorflow as tf
import sortedcontainers

class Agent():
    SUMMARIES = [
        ('rewards', tf.summary.histogram, tf.float32, [None]),
        ('run_time', tf.summary.scalar, tf.float32, []),
        ('final_score', tf.summary.scalar, tf.int32, []),
        ('points_per_second', tf.summary.scalar, tf.float32, []),
        ('death_image', tf.summary.image, tf.uint8, [1, config.params.image_size, config.params.image_size, 3]),
    ]

    def __init__(self, height: int, width: int, channels: int):
        '''Makes a game agent using make_model().'''

        tf.reset_default_graph()
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config_proto)
        self._file_writer = tf.summary.FileWriter(config.get_log_dir())

        self._height = height
        self._width = width
        self._channels = channels

        self._inputs, self._outputs, self._layers = model.make_model(1, height, width, channels)
        self._o_inputs, self._o_outputs, _ = model.make_optimizer(1, self._outputs['logits'])

        self._summaries: Dict[str, Any] = {}
        self._create_summaries()

        self._summaries_op = tf.summary.merge_all()
        self._sess.run(tf.global_variables_initializer())

        self._file_writer.add_graph(self._sess.graph)
        self._file_writer.flush()

        self._current_episode: List[Dict[str, Any]] = []
        self._episodes: List[pd.DataFrame] = []
        self._memory: pd.DataFrame = pd.DataFrame({'time': [], 'image': [], 'action': [], 'reward': []})
        self._start_time = time.time()

        self._sess.run(self._outputs['action'], feed_dict={
            self._inputs['images']: np.zeros([1, self._height, self._width, self._channels])
        })
        print('Ready!')

    def _create_summaries(self) -> None:
        for name, summary_fun, dtype, shape in Agent.SUMMARIES:
            self._summaries[name] = tf.placeholder(dtype, shape, name=f'{name}_summary__placeholder')
            summary_fun(name, self._summaries[name], family='game_stats')

    def randomize_model(self) -> None:
        self._sess.run(tf.global_variables_initializer())

    def start_new_game(self) -> None:
        self._start_time = time.time()
        self._current_episode = []

    def feed(self, image: np.ndarray) -> float:
        '''Evaluate the model with a new image.

        :param image: An np.ndarray of shape (height, width, channels)
        :returns: A float representing the probability that we should send OrbtXL a keydown'''

        t = time.time() - self._start_time
        logits, softmax, action = self._sess.run([self._outputs['logits'], self._outputs['softmax'], self._outputs['action']], feed_dict={
            self._inputs['images']: image.reshape((1, self._height, self._width, self._channels))
        })
        print(f'[{logits[0,0]:6.3f} {logits[0,1]:6.3f}] [{softmax[0,0]:6.3f} {softmax[0,1]:6.3f}] {action}')
        self._current_episode.append({
            'time': t,
            'image': image,
            'action': action,
            'reward': config.params.reward_nothing
        })

        return action == model.CATEGORIES['keydown']

    def collect(self, image: np.ndarray) -> float:
        '''Collect a new observation, and choose a random action.

        :param image: An np.ndarray of shape (height, width, channels)
        :returns: A float representing the probability that we should send OrbtXL a keydown'''

        t = time.time() - self._start_time
        action = random.choice([0, 1])
        self._current_episode.append({
            'time': t,
            'image': image,
            'action': action,
            'reward': config.params.reward_nothing
        })

        return action == model.CATEGORIES['keydown']

    def reward(self, amount: float) -> None:
        if self._current_episode:
            self._current_episode[-1]['reward'] += amount

    def gameover(self) -> None:
        if self._current_episode:
            self._current_episode[-1]['reward'] += config.params.reward_death

            # Discount all rewards
            # dicount ** (reward_discount_10db * fps) = 0.10
            fps = len(self._current_episode) / self._get_run_time()
            discount = 0.10 ** (1 / (config.params.reward_discount_10db * fps))
            g = 0.0
            for observation in reversed(self._current_episode):
                g = g * discount + observation['reward']
                observation['reward'] = g

            self._episodes.append(self._current_episode)
            self._memory = self._memory.append(self._current_episode, ignore_index=True)

    def _get_run_time(self) -> float:
        if self._current_episode:
            return self._current_episode[-1]['time'] - self._current_episode[0]['time']
        return 0

    def _get_death_image(self) -> np.ndarray:
        if self._current_episode:
            return self._current_episode[-1]['image'].reshape([1, config.params.image_size, config.params.image_size, 3])
        return np.zeros([1, config.params.image_size, config.params.image_size, 3])

    def train(self, final_score: int, pps: float) -> None:
        if self._memory.empty:
            return

        print(f'Training on {len(self._episodes)} episodes...')

        bins = np.arange(config.params.reward_death-5.1, config.params.reward_score+5.1, 1.0)
        reward_buckets = self._memory.groupby(pd.cut(self._memory['reward'], bins)).groups
        print({k: len(v) for k,v in reward_buckets.items()})

        sample = pd.DataFrame()
        min_size = len(min((i for i in reward_buckets.values() if not i.empty), key=len, default=[]))
        for bucket in reward_buckets.values():
            pop = self._memory.loc[bucket]
            if pop.empty:
                continue

            sample = sample.append(pop.sample(min(50, min(min_size, len(pop)))))

        print(f'Got {len(sample)} samples')

        first_time = True
        for _, row in sample.sample(frac=1).iterrows():
            if first_time:
                print('Dumping summaries for this batch...')
                summaries, step = self._sess.run([self._summaries_op, tf.train.get_global_step()], feed_dict={
                    self._summaries['rewards']: np.array([obs['reward'] for obs in self._current_episode]),
                    self._summaries['run_time']: self._get_run_time(),
                    self._summaries['final_score']: final_score,
                    self._summaries['points_per_second']: pps,
                    self._summaries['death_image']: self._get_death_image(),
                    self._inputs['images']: np.expand_dims(np.array(row['image']), 0),
                    self._o_inputs['actions']: np.expand_dims(row['action'], 0).astype(np.uint8),
                    self._o_inputs['rewards']: np.expand_dims(row['reward'], 0)
                })
                self._file_writer.add_summary(summaries, global_step=step)
                self._file_writer.flush()
                print('Summaries done, continuing training...')
            else:
                self._sess.run(self._o_outputs['optimizer'], feed_dict={
                    self._inputs['images']: np.expand_dims(np.array(row['image']), 0),
                    self._o_inputs['actions']: np.expand_dims(row['action'], 0).astype(np.uint8),
                    self._o_inputs['rewards']: np.expand_dims(row['reward'], 0)
                })

            first_time = False

    def close(self) -> None:
        '''Close the tf.Session'''

        self._file_writer.close()
        self._sess.close()