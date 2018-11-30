import datetime
import os.path

def get_log_dir() -> str:
    return os.path.join('tf_logs', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

monitor = 1  # TODO have user select this

episodes_dir = os.path.join('data', 'memory')
episode_strftime = '%Y-%m-%d_%H-%M-%S'

class params:
    image_size = 225  # Width and height of screenshots

    explore_rate = 0.40  # Probability of ignoring model and choosing a random action

    reward_nothing = -0.1  # Passive reward for accomplishing nothing but still not failing
    reward_score = 1  # Reward for scoring one OrbtXL point
    reward_death = -2  # Reward (penalty) for dying
    reward_discount_10db = 3.0  # Seconds to reach -10dB (10%) discount

    # [(filter_size, stride_size, output_depth)]
    pre_lstm_conv_layers = [
        (8, 4, 16, 'VALID'),
        (4, 2, 32, 'VALID')
    ]
    pre_lstm_fc_nodes = []
    state_size = 256
    post_lstm_fc_nodes = []

class training:
    learning_rate = 0.00005
    minimum_random_episodes = 4  # Minimum episodes to collect with random weights before we can train
    batch_size = 1
    max_checkpoints = 10
    checkpoint_dir = os.path.join('data', 'model')