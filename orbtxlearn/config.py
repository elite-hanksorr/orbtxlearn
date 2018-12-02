import datetime
import os.path

log_dir = os.path.join('data', 'logs')

def get_run_log_dir(run_type: str) -> str:
    return os.path.join(log_dir, datetime.datetime.now().strftime(f"{run_type}_%Y-%m-%d_%H-%M-%S"))

monitor = 1  # TODO have user select this

episodes_dir = os.path.join('data', 'memory')
episode_strftime = '%Y-%m-%d_%H-%M-%S'

class params:
    image_size = 480  # Width and height of screenshots

    explore_rate = 0.10  # Probability of ignoring model and choosing a random action

    rewards = {
        'nothing': -0.4,  # Passive reward for accomplishing nothing but still not failing
        'score': 2,  # Reward for scoring one OrbtXL point
        'death': -0.8,  # Reward (penalty) for dying
    }

    reward_discount_10db = 2.0  # Seconds to reach -10dB (10%) discount

    # [(filter_size, stride_size, output_depth)]
    pre_lstm_conv_layers = [
        (8, 4, 16, 'VALID'),
        (5, 2, 24, 'VALID'),
        (5, 2, 32, 'VALID')
    ]
    pre_lstm_fc_nodes = [1024]
    state_size = 256  # Size of LSTM. 0 for no LSTM
    post_lstm_fc_nodes = []

class training:
    learning_rate = 0.001
    minimum_random_episodes = 5  # Minimum episodes to collect with random weights before we can train
    batch_size = 50
    max_checkpoints = 5
    checkpoint_filename = os.path.join(log_dir, 'model')