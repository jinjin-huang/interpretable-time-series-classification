import argparse
import json
import os

DEFAULT_DATASET_NAME = "ArticularyWordRecognition"


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--experiment_run", type=str, default="001", help="Experiment run ID"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_prototypes", type=int, default=5, help="Number of prototypes per class"
    )
    parser.add_argument(
        "--feature_dim", type=int, default=128, help="Feature dimension after adapter"
    )
    parser.add_argument("--d_model", type=int, default=128, help="Transformer d_model")
    parser.add_argument(
        "--d_hidden", type=int, default=128, help="Transformer hidden dim"
    )
    parser.add_argument(
        "--q", type=int, default=8, help="Number of query heads per attention"
    )
    parser.add_argument(
        "--v", type=int, default=8, help="Number of value heads per attention"
    )
    parser.add_argument("--h", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--N", type=int, default=4, help="Number of Transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--pe", type=bool, default=True, help="Use positional encoding")
    parser.add_argument("--mask", type=bool, default=True, help="Use attention mask")
    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of experiment runs"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=21,
        help="Random seed for K-fold cross validation",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of epochs for training"
    )

    parser.add_argument(
        "--hyperparameter_tuning",
        type=int,
        default=0,
        help="Whether to perform hyperparameter tuning with K-fold cross validation",
    )

    args = parser.parse_args()
    args.hyperparameter_tuning = bool(args.hyperparameter_tuning)

    return args


def load_config(dataset_name):
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "dataset_configs.json")

    with open(config_path, "r") as f:
        configs = json.load(f)

    config = next(cfg for cfg in configs if cfg["dataset_name"] == dataset_name)
    return config


args = parse_args()

dataset_name = args.dataset_name
experiment_run = args.experiment_run

config = load_config(dataset_name)
num_classes = config["num_classes"]
num_dimensions = config["num_dimensions"]
data_path = config["data_path"]
data_length = config["data_length"]
train_file = f"{data_path}{dataset_name}_TRAIN.ts"
test_file = f"{data_path}{dataset_name}_TEST.ts"
