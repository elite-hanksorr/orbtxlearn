import datetime
import os.path

def get_log_dir() -> str:
    return os.path.join('tf_logs', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

class params:
    image_size = 480  # Width and height of screenshots

    reward_nothing = 0  # Passive reward for accomplishing nothing but still not failing
    reward_score = 1  # Reward for scoring one OrbtXL point
    reward_death = -2  # Reward (penalty) for dying
    reward_discount_10db = 3.0  # Seconds to reach -10dB (10%) discount

    # [(filter_size, stride_size, output_depth)]
    pre_lstm_conv_layers = [
        (8, 4, 12),
        (5, 2, 16),
        (4, 2, 20),
        (3, 2, 24)
    ]
    pre_lstm_fc_nodes = [1024]
    state_size = 256
    post_lstm_fc_nodes = [64, 16]

class training:
    learning_rate = 0.001
    minimum_random_episodes = 4  # Minimum episodes to collect with random weights before we can train
    batch_size = 1