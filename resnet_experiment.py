from experiment import Config, run

config = Config.from_yaml('configs/imagenet_resnet18.yaml')
run(config)