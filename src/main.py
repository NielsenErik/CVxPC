import os
import mlflow
import argparse
import json

from setup import set_up_mlflow
from experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path of the config file to use")
    parser.add_argument("--debug", action="store_true", help="Debug flag")
    args = parser.parse_args()
    config = json.load(open(args.config))
    set_up_mlflow(**config)
    with mlflow.start_run():
        run_experiment(config)

if __name__ == "__main__":
    main()