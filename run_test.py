import json
with open("src/config/pcnet.json", "r") as f:
    config = json.load(f)

config["execution_mode"] = "test"
config["epochs"] = 0

from src.experiment import run_experiment
run_experiment(config)
