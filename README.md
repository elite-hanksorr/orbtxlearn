# OrbtXLearn

## Developing

* CPU tensorflow: `pip install -e ".[cpu]"`
* GPU tensorflow (requires installation of CUDA libraries): `pip install -e ".[gpu]"`

## Installing

* CPU tensorflow: `pip install ".[cpu]"`
* GPU tensorflow (requires installation of CUDA libraries): `pip install ".[gpu]"`

## Running

The general strategy for running OrbtXLearn is to

1. (Optional) start `tensorboard --logdir=data/tf_logs` and navigate to `http://localhost:6006` in your browser
2. Start this project according to the instructions below
3. Wait for it to print out "Ready"
4. Start [orbtxlearn-spy][orbtxlearn-spy]: https://github.com/elite-hanksorr/orbtxlearn-spy
5. Hit "Play"

### Collecting data

Currently the only ways to collect data are to let OrbtXLearn play completely randomly,
or by having it play using the current model. In the future, you should be able to
collect human input.

    python -m orbtxlearn run [--host HOST] [--port PORT] [--no-model | --model] [--restore-model | --no-restore-model]

### Training

    python -m orbtxlearn train [--restore-model|--no-restore-model] EPOCHS

### Running with the trained model

    python -m orbtxlearn run [--host HOST] [--port PORY] --model --restore-model
