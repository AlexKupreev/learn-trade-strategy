"""The main script for strategy estimation.

It works with the model already designed on exploratory stage (here it's RandomForest).
"""

import argparse
import tomllib


from scripts.data_repo import DataRepository
from scripts.transform import TransformData
from scripts.train import SimulationParams, TrainModel


def main(
    fetch_repo: bool,
    transform_data: bool,
    train_model: bool,
    config: str,
    data_folder: str,
):
    """The main entrypoint into strategy estimation"""
    sim_params = load_config(config)

    print("Step 1: Getting data from APIs or Load from disk")
    repo = DataRepository()

    if fetch_repo:
        print("Fetching data from APIs...")

        # Fetch All 3 datasets for all dates from APIs
        repo.fetch()

        print("Saving data to disk...")
        repo.persist(data_dir=data_folder)
    else:
        print("Loading data from disk...")

        repo.load(data_dir=data_folder)

    print("Step2: Transform data into one dataframe")
    transformed = TransformData(repo=repo)

    if transform_data:
        print("Preparing data for training/inference...")
        transformed.transform()

        print("Saving data to disk...")
        transformed.persist(data_dir=data_folder)
    else:
        print("Loading data from disk...")
        transformed.load(data_dir=data_folder)

    print("Step3: Train/Load Model")

    trained = TrainModel(transformed=transformed)

    if train_model:
        print("Prepare dataframe for training...")
        trained.prepare_dataframe()

        print("Training the model...")
        trained.train_random_forest()

        print("Saving the model to disk...")
        trained.persist(data_dir=data_folder)
    else:
        print("Prepare dataframe for training...")
        trained.prepare_dataframe()

        print("Loading the model from disk...")
        trained.load(data_dir=data_folder)

    print("Step4: Inference")
    trained.make_inference(sim_params)

    print("Results of the estimation (last 10):")
    predicted_signals = trained.get_last_signals(num=10)

    print(predicted_signals.tail(10))

    print("Simulation results:")
    res, capital = trained.simulate(sim_params)


def load_config(file_path):
    """Load configuration settings from a TOML file using tomllib."""
    with open(file_path, "rb") as file:
        data = tomllib.load(file)
    return SimulationParams(**data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A strategy trading script.")

    # Adding boolean flag arguments
    parser.add_argument(
        "--fetch-repo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, full data load will be done, if not set (default) - existing file from disk will be loaded.",
    )
    parser.add_argument(
        "--transform-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, data will be transformed into a single dataset, "
        "if no (default), the dataset will be loaded from the local storage",
    )
    parser.add_argument(
        "--train-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, the model will be trained, if no (default), "
        "the model will be loaded from the local storage",
    )
    parser.add_argument(
        "--config", type=str, default="config.toml", help="Path to configuration file"
    )
    parser.add_argument(
        "--data-folder", type=str, default="data/", help="Path to data folder"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Pass the values to the main function
    main(
        args.fetch_repo,
        args.transform_data,
        args.train_model,
        args.config,
        args.data_folder,
    )
