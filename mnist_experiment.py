from experiment import Config, run

config = Config.from_yaml('configs/mnist_cnn.yaml')
run(config)