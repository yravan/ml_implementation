"""
Runnable experiment script for PyCharm debugging.
Change CONFIG_PATH below, then Run/Debug this file.
"""
from experiment import Config, run

CONFIG_PATH = "configs/translation_multi30k_mac.yaml"

if __name__ == '__main__':
    config = Config.from_yaml(CONFIG_PATH)
    run(config)
