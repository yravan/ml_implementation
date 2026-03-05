"""
Runnable experiment script for PyCharm debugging.
Change CONFIG_PATH below, then Run/Debug this file.
"""
from experiment import Config, run

CONFIG_PATH = "configs/gpt2_wikitext2_mac.yaml"

config = Config.from_yaml(CONFIG_PATH)
run(config)
