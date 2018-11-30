import datetime
import os.path
import random
import sqlite3
import time
from typing import List, Union, Dict, Any, MutableSequence

from . import model, config, util

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import sortedcontainers

util.sqlite_register_ndarray()

class Agent():
    SUMMARIES = [
        ('rewards', tf.summary.histogram, tf.float32, [None]),
        ('run_time', tf.summary.scalar, tf.float32, []),
        ('final_score', tf.summary.scalar, tf.int32, []),
        ('points_per_second', tf.summary.scalar, tf.float32, []),
        ('death_image', tf.summary.image, tf.uint8, [1, config.params.image_size, config.params.image_size, 3]),
    ]

    def __init__(self, height: int, width: int, channels: int, save_episodes: bool = True):
        '''Makes a game agent using make_model().'''

        tf.reset_default_graph()
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config_proto)
        self._file_writer = tf.summary.FileWriter(config.get_log_dir())

        # FIXME should we really be referring to files outside the package?
        self._db_conn = sqlite3.connect(os.path.join(config.episodes_dir, 'metadata.sqlite'), detect_types=sqlite3.PARSE_DECLTYPES)
        self._db_conn.row_factory = util.dict_factory
        with self._db_conn:
            self._db_conn.execute(
                '''
                create table if not exists Observations (
                    id integer primary key autoincrement,
                    filename text not null,
                    episode_time real not null,
                    image ndarray not null,
                    action integer not null,
                    reward real not null
                )
                ''')

        self._height = height
        self._width = width
        self._channels = channels

        self._inputs, self._outputs, self._layers = model.make_model(1, height, width, channels)
        self._optim_inputs, self._optim_outputs, _ = model.make_optimizer(1, self._outputs['logits'])

        self._summaries: Dict[str, Any] = {}
        self._create_summaries()

        self._all_summaries_op = tf.summary.merge_all()
        self._sess.run(tf.global_variables_initializer())

        self._file_writer.add_graph(self._sess.graph)
        self._file_writer.flush()

        self._saving_episodes = save_episodes
        self._current_episode: List[Dict[str, Any]] = []
        self._start_time = datetime.datetime.now()

        self._sess.run(self._outputs['action'], feed_dict={
            self._inputs['images']: np.zeros([1, self._height, self._width, self._channels])
        })
        print('Ready!')

    def _create_summaries(self) -> None:
        with tf.variable_scope('game_stats'):
            for name, summary_fun, dtype, shape in Agent.SUMMARIES:
                self._summaries[name] = tf.placeholder(dtype, shape, name=f'{name}_summary_placeholder')
                summary_fun(name, self._summaries[name])

    def _save_episode(self, episode: List[Dict[str, Any]]) -> None:
        print('Saving episode to database...')

        timestr = self._start_time.strftime(config.episode_strftime)

        # Make new directory
        i = 0
        while True:
            try:
                if i == 0:
                    dirname = os.path.join(config.episodes_dir, timestr)
                else:
                    dirname = os.path.join(config.episodes_dir, timestr + f'_{i}')

                os.mkdir(dirname)
            except FileExistsError:
                i += 1
            else:
                break

        for i, obs in enumerate(self._current_episode):
            obs['filename'] = os.path.join(dirname, f'{i:05}.png')
            Image.fromarray(obs['image'], mode='RGB').save(obs['filename'])

        with self._db_conn:
            self._db_conn.executemany(
                '''
                insert into Observations (episode_time, filename, image, action, reward)
                values (:episode_time, :filename, :image, :action, :reward)
                ''', self._current_episode)


    def randomize_model(self) -> None:
        self._sess.run(tf.global_variables_initializer())

    def start_new_game(self) -> None:
        self._start_time = datetime.datetime.now()
        self._current_episode = []

    def feed(self, image: np.ndarray) -> float:
        '''Evaluate the model with a new image.

        :param image: An np.ndarray of shape (height, width, channels)
        :returns: A float representing the probability that we should send OrbtXL a keydown'''

        start_time = time.time()
        episode_time = (datetime.datetime.now() - self._start_time).total_seconds()
        logits, softmax, action = self._sess.run([self._outputs['logits'], self._outputs['softmax'], self._outputs['action']], feed_dict={
            self._inputs['images']: image.reshape((1, self._height, self._width, self._channels))
        })
        elapsed = time.time() - start_time

        action = int(action)
        if random.random() < config.params.explore_rate:
            action = random.randint(0, 1)

        print(f'[{logits[0,0]:6.3f} {logits[0,1]:6.3f}] [{softmax[0,0]:6.3f} {softmax[0,1]:6.3f}] {action} {1000*(time.time()-start_time):6.1f}ms')

        self._current_episode.append({
            'episode_time': episode_time,
            'filename': None,
            'image': image,
            'action': action,
            'reward': config.params.reward_nothing
        })

        return action == model.CATEGORIES['keydown']

    def collect(self, image: np.ndarray) -> float:
        '''Collect a new observation, and choose a random action.

        :param image: An np.ndarray of shape (height, width, channels)
        :returns: A float representing the probability that we should send OrbtXL a keydown'''

        t = (datetime.datetime.now() - self._start_time).total_seconds()
        action = random.randint(0, 1)
        self._current_episode.append({
            'episode_time': t,
            'filename': None,
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

            if self._saving_episodes:
                self._save_episode(self._current_episode)

    def _get_run_time(self) -> float:
        if self._current_episode:
            return self._current_episode[-1]['episode_time'] - self._current_episode[0]['episode_time']
        return 0

    def _get_death_image(self) -> np.ndarray:
        if self._current_episode:
            return self._current_episode[-1]['image'].reshape([1, config.params.image_size, config.params.image_size, 3])
        return np.zeros([1, config.params.image_size, config.params.image_size, 3])

    def train(self, final_score: int, pps: float) -> None:
        memory_len = self._db_conn.execute('select count(*) as count from Observations').fetchone()['count']
        if not memory_len:
            print('No samples retrieved from database, not training')
            return

        print(f'Training on {memory_len} samples...')

        buckets: List[Dict[str, int]] = self._db_conn.execute(
            '''
            select round(reward, 1) as bin, count(*) as count
            from Observations
            group by bin
            order by bin
            ''').fetchall()
        print(buckets)

        sample: List[Dict[str, Any]] = []
        sizes = sorted(b['count'] for b in buckets)
        min_size = sizes[len(sizes)//5] * 3 # Lowest quintile
        print(f'Using minsize = {min_size}')
        for bucket in buckets:
            sample.extend(
                self._db_conn.execute(
                    '''
                    select image, action, reward
                    from Observations
                    where round(reward, 1) = ?
                    limit ?
                    ''', (bucket['bin'], min(50, min_size))).fetchall())

        print(f'Got {len(sample)} samples')
        random.shuffle(sample)

        training_summary = tf.summary.merge([
            tf.summary.merge_all(scope='train'),
            tf.summary.merge_all(scope='model')
        ])
        game_stats_summary = tf.summary.merge_all(scope='game_stats')

        first_time = True
        for row in sample:
            if first_time:
                summary, step = self._sess.run(
                    [game_stats_summary, tf.train.get_global_step()],
                    feed_dict={
                        self._summaries['rewards']: np.array([obs['reward'] for obs in self._current_episode]),
                        self._summaries['run_time']: self._get_run_time(),
                        self._summaries['final_score']: final_score,
                        self._summaries['points_per_second']: pps,
                        self._summaries['death_image']: self._get_death_image(),
                    })

                self._file_writer.add_summary(summary, global_step=step)
                first_time = False

            _, summary, step = self._sess.run(
                [self._optim_outputs['optimizer'], training_summary, tf.train.get_global_step()],
                feed_dict={
                    self._inputs['images']: np.expand_dims(row['image'], 0),
                    self._optim_inputs['actions']: np.expand_dims(row['action'], 0).astype(np.uint8),
                    self._optim_inputs['rewards']: np.expand_dims(row['reward'], 0)
                })

            self._file_writer.add_summary(summary, global_step=step)

        self._file_writer.flush()

    def close(self) -> None:
        '''Close the tf.Session'''

        self._file_writer.close()
        self._sess.close()