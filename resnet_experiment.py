from experiment import Config, run

config = Config.from_yaml('configs/imagenette_resnet.yaml')
run(config)