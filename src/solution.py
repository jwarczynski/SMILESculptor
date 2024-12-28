import argparse
import pickle
import yaml

import torch

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateFinder, ModelCheckpoint, BatchSizeFinder
from lightning.pytorch.loggers import WandbLogger

from pprint import pprint


from src.SmilesVectorizer import SmilesVectorizer
from src.data_loader import create_data_module
from src.models import *


def load_config(file_path):
    """Load YAML configuration file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def instantiate_class(class_name, class_args):
    """Dynamically instantiate a class by name."""
    cls = globals().get(class_name)
    if cls is None:
        raise ValueError(f"Class {class_name} not found.")
    return cls(**class_args)


def call_function(function_name, function_args):
    """Dynamically call a function by name."""
    func = globals().get(function_name)
    if func is None:
        raise ValueError(f"Function {function_name} not found.")
    return func(**function_args)


def parse_args():
    parser = argparse.ArgumentParser(description="Run model and data module setup based on a YAML configuration.")

    # General arguments
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--loss", type=str, required=True, choices=["bce", "ce"],
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Name of the run (required)."
    )
    parser.add_argument(
        "--wandb-run-id", type=str, default=None,
        help="Specify the WandB run ID to resume from. If not provided, a new run will be created."
    )
    parser.add_argument(
        "--project-name", type=str, default="MolsVAE",
        help="Project name for the WandB logger (default: MolsVAE)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--tags", type=str, nargs='*', default=[],
        help="Tags for the WandB run (space-separated)."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for the data loader (default: None, will use value from config if not specified)."
    )
    parser.add_argument(
        "--moles-path", type=str, default=None,
        help="Path to the data file (default: None, will use value from config if not specified)."
    )
    parser.add_argument(
        "--int-to-char-path", type=str, default="data/itc_500k.pkl",
        help="Path to the 'int_to_char.pkl' file (default: None, will use value from config if not specified)."
    )
    parser.add_argument(
        "--vectorizer-path", type=str, default="data/sv_no_stereo_500.pkl",
        help="Path to dumped vectorizer file (default: None)."
    )

    # Trainer arguments with default values
    parser.add_argument(
        "--max-epochs", type=int, default=2000,
        help="Maximum number of epochs (default: 2000)."
    )
    parser.add_argument(
        "--log-every-n-steps", type=int, default=1,
        help="Log every N steps (default: 1)."
    )
    parser.add_argument(
        "--num-sanity-val-steps", type=int, default=0,
        help="Number of sanity validation steps (default: 0)."
    )
    parser.add_argument(
        "--limit-train-batches", type=float, default=1.0,
        help="Limit the proportion of training batches (default: 1.0)."
    )
    parser.add_argument(
        "--limit-val-batches", type=float, default=1.0,
        help="Limit the proportion of validation batches (default: 1.0)."
    )
    parser.add_argument(
        "--limit-test-batches", type=float, default=1.0,
        help="Limit the proportion of test batches (default: 1.0)."
    )
    parser.add_argument(
        "--enable-progress-bar", type=lambda x: str(x).lower() in ("true", "1"),
        default=True,
        help="Enable progress bar (default: True)."
    )
    parser.add_argument(
        "--enable-checkpointing", type=lambda x: str(x).lower() in ("true", "1"),
        default=True,
        help="Enable checkpointing (default: True)."
    )
    parser.add_argument(
        "--enable-model-summary", type=bool, default=True,
        help="Enable model summary (default: True)."
    )
    parser.add_argument(
        "--logger", type=str, default="wandb_logger",
        help="Logger for the training (default: wandb_logger)."
    )
    parser.add_argument(
        "--deterministic", type=bool, default=True,
        help="Set deterministic training (default: True)."
    )
    parser.add_argument(
        "--accelerator", type=str, default="auto",
        help="The accelerator to use (default: gpu)."
    )
    parser.add_argument(
        "--devices", type=int, default=4,
        help="Number of devices (default: 4)."
    )
    parser.add_argument(
        "--overfit-batches", type=int, default=0,
        help="Proportion of data to overfit (default: 0)."
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default=None,
        help="Path to a checkpoint file for resuming training (default: None)."
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=None,
        help="Number of epochs with no improvement after which training will be stopped (default: None)."
    )
    parser.add_argument(
        "--early-stopping-monitor", type=str, default="val/binary_ce_recon_loss",
        help="Metric to monitor for early stopping (default: 'val/bce_loss')."
    )
    parser.add_argument(
        "--early-stopping-mode", type=str, default="min", choices=["min", "max"],
        help="Mode for early stopping ('min' or 'max', default: 'min')."
    )

    # Tensor Core precision argument
    parser.add_argument(
        "--matmul-precision", type=str, choices=["highest", "high", "medium"],
        default="high",  # Default to 'highest' for balanced performance
        help="Set the precision for Tensor Core matmul operations ('highest', 'high', or 'medium'). Default: 'highest'."
    )

    return parser.parse_args()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    args = parse_args()
    pprint(vars(args))

    torch.set_float32_matmul_precision(args.matmul_precision)

    # Set seed
    seed_everything(args.seed, workers=True)

    # Load the YAML configuration file
    config = load_config(args.config)

    data_module_config = config.get("data_module", {})

    # Overwrite values from the command line if specified
    if args.batch_size is not None:
        data_module_config["args"]["batch_size"] = args.batch_size

    if args.moles_path is not None:
        data_module_config["args"]["path"] = args.moles_path

    print(f'moles path: {data_module_config["args"]["path"]}')
    data_module_result = call_function(data_module_config["function"], data_module_config["args"])
    data_module, max_len, _ = data_module_result

    if args.vectorizer_path is not None:
        pprint(f'vectorizer path: {args.vectorizer_path}')
        sv = pickle.load(open(args.vectorizer_path, 'rb'))
        int_to_char = sv.int_to_char

    # Extract dynamically computed values for model args
    calculated_args = {
        "seq_len": max_len,
        "charset_size": len(int_to_char),
        "int_to_char": int_to_char
    }
    pprint(f'calculated args: {calculated_args}')

    # Merge calculated args with YAML-configured args for the model
    model_config = config.get("model", {})
    model_args = {**model_config["args"], **calculated_args, "loss": args.loss}

    # Instantiate the model
    model = instantiate_class(model_config["name"], model_args)
    print(model.model)

    wandb_logger = WandbLogger(
        project=args.project_name,
        name=args.run_name,
        log_model=True,
        config=model_config,
        tags=args.tags,
        id=args.wandb_run_id,  # Use the provided run ID or None
        resume="allow" if args.wandb_run_id else None,  # Resume if run ID is provided
    )
    wandb_logger.watch(model, log="all")

    callbacks = [
        # LearningRateFinder(min_lr=1e-6, max_lr=1e-2),
        # BatchSizeFinder()
    ]

    if args.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                monitor=args.early_stopping_monitor, save_top_k=3, mode=args.early_stopping_mode,
                dirpath=f'checkpoints/{args.run_name}',
                filename=f'epoch={{epoch:02d}}-step={{step}}-loss={{{args.early_stopping_monitor}:.2f}}',
                save_last=True,
                auto_insert_metric_name=False
            )
        )

    if args.early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor=args.early_stopping_monitor,
                patience=args.early_stopping_patience,
                mode=args.early_stopping_mode
            )
        )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        enable_progress_bar=args.enable_progress_bar,
        enable_checkpointing=args.enable_checkpointing,
        enable_model_summary=args.enable_model_summary,
        logger=wandb_logger,
        deterministic=args.deterministic,
        accelerator=args.accelerator,
        devices=args.devices,
        overfit_batches=args.overfit_batches,
        callbacks=callbacks,  # Use the updated callbacks list
    )

    trainer.fit(model, data_module, ckpt_path=args.checkpoint_path)
    trainer.test(model, data_module)
