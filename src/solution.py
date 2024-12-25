import argparse
import os
import sys
import pickle
import importlib

import exca
import pydantic
import yaml

from pydantic import BaseModel, Field, FilePath
from typing import Optional, List

from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateFinder, ModelCheckpoint, BatchSizeFinder
from lightning.pytorch.loggers import WandbLogger

from pprint import pprint

from src.SmilesVectorizer import SmilesVectorizer
from src.data_loader import create_data_module
from src.models import *
import logging

logging.getLogger("exca").setLevel(logging.DEBUG)


def load_config(file_path):
    """Load YAML configuration file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def instantiate_class(class_name, class_args):
    """
    Dynamically instantiate a class by name.

    Args:
        class_name (str): The name of the class, optionally in 'module.ClassName' format.
        class_args (dict): The arguments to pass to the class constructor.

    Returns:
        object: An instance of the specified class.

    Raises:
        ValueError: If the class cannot be found or instantiated.
    """
    # Split the module and class name if provided in 'module.ClassName' format
    if '.' in class_name:
        module_name, cls_name = class_name.rsplit('.', 1)
        try:
            # Dynamically import the module
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Class {class_name} not found: {e}")
    else:
        # Use globals() as a fallback if no module is provided
        cls = globals().get(class_name)
        if cls is None:
            raise ValueError(f"Class {class_name} not found in globals.")

    # Instantiate the class with the provided arguments
    try:
        return cls(**class_args)
    except TypeError as e:
        raise ValueError(f"Error instantiating {class_name}: {e}")


def call_function(function_name, function_args):
    """Dynamically call a function by name."""
    # Split the module and function name if provided in 'module.function' format
    if '.' in function_name:
        module_name, func_name = function_name.rsplit('.', 1)
        try:
            # Dynamically import the module
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Function {function_name} not found: {e}")
    else:
        # Use globals() as a fallback if no module is provided
        func = globals().get(function_name)
        if func is None:
            raise ValueError(f"Function {function_name} not found in globals.")

    # Call the function with the provided arguments
    return func(**function_args)


def run(config):
    # Set seed
    seed_everything(config.model.seed)

    # Set the precision for Tensor Core matmul operations
    torch.set_float32_matmul_precision(config.trainer.matmul_precision)

    # Load the YAML configuration file
    model_config = load_config(config.model.config)

    data_module_config = model_config.get("data_module", {})
    pprint(data_module_config)

    # Overwrite values from the command line if specified
    if config.data is not None:
        data_module_config["args"]["batch_size"] = config.data.batch_size

    if config.data.moles_path is not None:
        data_module_config["args"]["path"] = config.data.moles_path

    print(f'moles path: {data_module_config["args"]["path"]}')
    data_module_result = call_function("src.data_loader." + data_module_config["function"], data_module_config["args"])
    data_module, max_len, _ = data_module_result

    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if src_path not in sys.path:
        print("appending src path")
        sys.path.append(src_path)

    if config.data.vectorizer_path is not None:
        pprint(f'vectorizer path: {config.data.vectorizer_path}')
        sv = pickle.load(open(config.data.vectorizer_path, 'rb'))
        int_to_char = sv.int_to_char

    # Extract dynamically computed values for model args
    calculated_args = {
        "seq_len": max_len,
        "charset_size": len(int_to_char),
        "int_to_char": int_to_char
    }
    pprint(f'calculated args: {calculated_args}')

    # Merge calculated args with YAML-configured args for the model
    model_config = model_config.get("model", {})
    model_args = {**model_config["args"], **calculated_args, "loss": config.model.loss}

    # Instantiate the model
    model = instantiate_class("src.models." + model_config["name"], model_args)
    print(model.model)

    wandb_logger = WandbLogger(
        **config.wandb.dict(),
        log_model=True,
        resume="allow" if config.wandb.id else None,  # Resume if run ID is provided
    )
    wandb_logger.watch(model, log="all")

    callbacks = [
        # LearningRateFinder(min_lr=1e-6, max_lr=1e-2),
        # BatchSizeFinder()
    ]

    if config.trainer.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                monitor=config.trainer.early_stopping_monitor, save_top_k=3,
                mode=config.trainer.early_stopping_mode,
                dirpath=f'checkpoints/{config.wandb.name}',
                filename=f'epoch={{epoch:02d}}-step={{step}}-loss={{{config.trainer.early_stopping_monitor}:.2f}}',
                save_last=True,
                auto_insert_metric_name=False
            )
        )

    if config.trainer.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=config.trainer.early_stopping_monitor,
                patience=config.trainer.early_stopping_patience,
                mode=config.trainer.early_stopping_mode
            )
        )

    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        limit_test_batches=config.trainer.limit_test_batches,
        enable_progress_bar=config.trainer.enable_progress_bar,
        enable_checkpointing=config.trainer.enable_checkpointing,
        enable_model_summary=config.trainer.enable_model_summary,
        logger=wandb_logger,
        deterministic=config.trainer.deterministic,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        overfit_batches=config.trainer.overfit_batches,
        callbacks=callbacks,  # Use the updated callbacks list
    )

    trainer.fit(model, data_module, ckpt_path=config.trainer.checkpoint_path)
    trainer.test(model, data_module)


# Pydantic model for model configuration
class ModelConfig(BaseModel):
    config: FilePath = Field(..., description="Path to the YAML configuration file.")
    loss: str = Field(..., description="Loss function type.", enum=["bce", "ce"])
    seed: int = Field(42, description="Random seed for reproducibility.")
    model_config = pydantic.ConfigDict(extra="forbid")


# Pydantic model for data paths
class DataConfig(BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    batch_size: Optional[int] = Field(None, description="Batch size for the data loader.")
    moles_path: Optional[FilePath] = Field(None, description="Path to the data file.")
    int_to_char_path: FilePath = Field("data/itc_500k.pkl", description="Path to the 'int_to_char.pkl' file.")
    vectorizer_path: FilePath = Field("data/sv_no_stereo_500.pkl", description="Path to the dumped vectorizer file.")


# Pydantic model for WandB configuration
class WandbConfig(BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    project: str = Field("MolsVAE", description="Project name for WandB.")
    name: str = Field(..., description="Name of the run.")
    id: Optional[str] = Field(None, description="WandB run ID to resume from.")
    tags: List[str] = Field(default_factory=list, description="Tags for the WandB run.")


# Pydantic model for trainer configuration
class TrainerConfig(BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    max_epochs: int = Field(100, description="Maximum number of epochs.")
    log_every_n_steps: int = Field(1, description="Log every N steps.")
    num_sanity_val_steps: int = Field(0, description="Number of sanity validation steps.")
    limit_train_batches: float = Field(1.0, description="Limit the proportion of training batches.")
    limit_val_batches: float = Field(1.0, description="Limit the proportion of validation batches.")
    limit_test_batches: float = Field(1.0, description="Limit the proportion of test batches.")
    enable_progress_bar: bool = Field(True, description="Enable progress bar.")
    enable_checkpointing: bool = Field(True, description="Enable checkpointing.")
    enable_model_summary: bool = Field(True, description="Enable model summary.")
    logger: str = Field("wandb_logger", description="Logger for the training.")
    deterministic: bool = Field(True, description="Set deterministic training.")
    accelerator: str = Field("auto", description="The accelerator to use.")
    devices: int = Field(4, description="Number of devices.")
    overfit_batches: int = Field(0, description="Proportion of data to overfit.")
    checkpoint_path: Optional[FilePath] = Field(None, description="Path to a checkpoint file.")
    early_stopping_patience: Optional[int] = Field(None, description="Number of epochs for early stopping.")
    early_stopping_monitor: str = Field("val/binary_ce_recon_loss", description="Metric to monitor for early stopping.")
    early_stopping_mode: str = Field("min", description="Mode for early stopping.", enum=["min", "max"])
    matmul_precision: str = Field("high", description="Tensor Core matmul precision.",
                                  enum=["highest", "high", "medium"])


# Composite model for overall configuration
class ExperimentConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    trainer: TrainerConfig
    wandb: WandbConfig
    infra: exca.TaskInfra = exca.TaskInfra(
        folder="jobs",
        keep_in_ram=False,
        mode="force",
        cluster="local",
        cpus_per_task=4,
        conda_env="temp",
        workdir={
            "copied": ["tox.ini"],
            "folder": "jobs"
        }
    )

    @infra.apply()
    def run(self):
        import sys
        sys.path.append(os.getcwd())

        run(self)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model and data module setup based on a YAML configuration.")
    parser.add_argument("--experiment", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    copied = (
            ["data", "src", "configs"] +
            [f"src/{file}" for file in os.listdir("src")] +
            [f"data/{file}" for file in os.listdir("data")] +
            [f"configs/{file}" for file in os.listdir("configs")]
    )

    exp_config = load_config(args.experiment)
    experiment = ExperimentConfig(**exp_config, infra=exca.TaskInfra(
        folder="jobs",
        keep_in_ram=False,
        mode="force",
        cluster="slurm",
        cpus_per_task=4,
        conda_env="temp",
        workdir={
            "copied": copied,
            "includes": ["*.py", "*.yml", "*.pkl", "*.npy"],
            # "folder": "jobs"
        }
    ))

    losses = ["bce", "ce"]
    data = [
        ("data/moles_ohe_no_stereo_sv_500k.npy", "data/sv_no_stereo_500.pkl"),
        ("data/moles_ohe_500k.npy", "data/vec_ohe_500k.pkl")
    ]
    run_names = ["bce_no_stereo", "ce_no_stereo", "bce_stereo", "ce_stereo"]

    import itertools

    with experiment.infra.job_array() as arr:
        for i, (loss, (moles_path, vectorizer_path)) in enumerate(itertools.product(losses, data)):
            arr.append(experiment.infra.clone_obj({
                "model": {
                    "loss": loss
                },
                "data": {
                    "moles_path": moles_path,
                    "vectorizer_path": vectorizer_path
                },
                "wandb": {
                    "name": run_names[i]
                }
            }))

    print(arr[0].infra.job())
