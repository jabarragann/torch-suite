from abc import ABC, abstractclassmethod
import json
from pathlib import Path
from re import I
from tabnanny import check
import numpy as np
import time
from rich.progress import track
from dataclasses import dataclass, field
from typing import Callable, List

# Torch
import optuna
from sklearn.multioutput import MultiOutputClassifier
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# Custom
from torchsuite.utils.Logger import Logger
from pytorchcheckpoint.checkpoint import CheckpointHandler

log = Logger(__name__).log
np.set_printoptions(precision=4, suppress=True, sign=" ")


@dataclass
class EndOfEpochMetric:
    name: str
    checkpoint_handler: CheckpointHandler
    function: Callable
    loader: DataLoader

    def __post_init__(self):
        self.current_val: float = 0.0

    def calc_new_value(self, epoch=None):
        if epoch is None:
            raise Exception("epoch cannot be None")
        self.current_val = self.function(self.loader)
        self.checkpoint_handler.store_running_var(var_name=self.name, iteration=epoch, value=self.current_val)


class BestModelSaver:
    def __init__(self, direction: str, metric_name: str) -> None:
        assert direction in ["maximize", "minimize"], "Not valid direction"
        assert metric_name in ["train_loss", "train_acc", "valid_acc"], "Not valid metric name"

        self.direction = direction
        self.metric_name = metric_name
        self.best_metric = -99999.0 if direction == "maximize" else 99999.0

    def __call__(self, checkpoint_handler: CheckpointHandler = None, save_function: callable = None):
        if checkpoint_handler is None or save_function is None:
            Exception("error. wrong input parameters")

        current_metric_dict = checkpoint_handler.__getattribute__(self.metric_name)
        last_key = list(current_metric_dict)[-1]
        current_metric = current_metric_dict[last_key]
        if self.direction == "maximize":
            if current_metric > self.best_metric:
                save_function("best_checkpoint.pt")
                self.best_metric = current_metric
                log.info(f"Save best model at epoch {last_key}: {current_metric:0.06f}")
        elif self.direction == "minimize":
            if current_metric < self.best_metric:
                save_function("best_checkpoint.pt")
                self.best_metric = current_metric
                log.info(f"Save best model at epoch {last_key}: {current_metric:0.06f}")

        return self.best_metric


@dataclass
class Trainer(ABC):
    train_loader: DataLoader
    valid_loader: DataLoader
    net: nn.Module
    optimizer: nn.Module
    loss_metric: nn.Module
    epochs: int
    root: Path
    gpu_boole: bool = True
    optimize_hyperparams: bool = False
    save: bool = True
    log_interval: int = 1
    end_of_epoch_metrics: List = field(default_factory=lambda: ["train_acc", "valid_acc"])
    metric_to_opt: str = "train_loss"
    direction_to_opt: str = "minimize"

    def __post_init__(self):
        self.init_epoch = 0
        self.final_epoch = 0
        self.batch_count = 0
        self.checkpoint_handler = CheckpointHandler()
        self.best_model_saver = BestModelSaver(self.direction_to_opt, self.metric_to_opt)
        self.best_metric_to_opt = self.best_model_saver.best_metric

        self.epoch_metrics_dict = {}
        for m in self.end_of_epoch_metrics:
            assert m in ["train_acc", "valid_acc"], "only train_acc or valid_acc in end of epoch metrics"
            data_l = m.strip().split("_")[0]
            metric = EndOfEpochMetric(
                name=m,
                checkpoint_handler=self.checkpoint_handler,
                function=self.calculate_acc,
                loader=self.__getattribute__(data_l + "_loader"),
            )
            self.epoch_metrics_dict[m] = metric

    def train_loop(self, trial: optuna.Trial = None, verbose=True):
        log.info(f"Starting Training")
        valid_acc = 0
        for epoch in track(range(self.init_epoch, self.epochs), "Training network"):
            time1 = time.time()
            loss_sum = 0
            total = 0

            # Batch loop
            local_loss_sum = 0
            local_total = 0
            for batch_idx, (x, y) in enumerate(self.train_loader):
                if self.gpu_boole:
                    x = x.cuda()
                    y = y.cuda()

                # loss calculation and gradient update:
                self.optimizer.zero_grad()
                outputs = self.net(x)
                loss = self.loss_metric(outputs, y)  # REMEMBER loss(OUTPUTS,LABELS)
                loss.backward()
                self.optimizer.step()  # Update parameters

                self.batch_count += 1
                # Global loss
                loss_sum += loss * y.shape[0]
                total += y.shape[0]
                # Local loss
                local_loss_sum += loss * y.shape[0]
                local_total += y.shape[0]

                if batch_idx % self.log_interval == 0:
                    local_loss = (local_loss_sum / local_total).cpu().item()
                    # End of batch stats
                    self.checkpoint_handler.store_running_var(
                        var_name="train_loss_batch", iteration=self.batch_count, value=local_loss
                    )
                    if verbose:
                        log.info(
                            f"epoch {epoch:3d} batch_idx {batch_idx:4d}/{len(self.train_loader)-1} local_loss {local_loss:0.6f}"
                        )

                    local_loss_sum = 0
                    local_total = 0

            # End of epoch statistics
            train_loss = loss_sum / total
            train_loss = train_loss.cpu().item()
            self.checkpoint_handler.store_running_var(var_name="train_loss", iteration=epoch, value=train_loss)

            # End of epoch additional metrics
            for m in self.end_of_epoch_metrics:
                self.epoch_metrics_dict[m].calc_new_value(**{"epoch": epoch})

            self.final_epoch = epoch
            self.init_epoch = self.final_epoch

            # Saving models
            if self.save:
                self.best_metric_to_opt = self.best_model_saver(self.checkpoint_handler, self.save_checkpoint)
            # if "valid_acc" in self.epoch_metrics_dict:
            #     valid_acc = self.epoch_metrics_dict["valid_acc"].current_val
            #     if valid_acc > self.best_valid_acc and self.save:
            #         log.info("saving best validation model")
            #         self.best_valid_acc = valid_acc
            #         self.save_checkpoint("best_checkpoint.pt")
            if self.save:
                self.save_checkpoint("final_checkpoint.pt")

            # Print epoch information
            time2 = time.time()
            if verbose:
                log.info(f"*" * 30)
                log.info(f"Epoch {epoch}/{self.epochs-1}:")
                log.info(f"Elapsed time for epoch: { time2 - time1:0.04f} s")
                log.info(f"Training loss:     {train_loss:0.8f}")
                for m in self.end_of_epoch_metrics:
                    log.info(f"{m}: {self.epoch_metrics_dict[m].current_val:0.06f}")
                log.info(f"*" * 30)

            # Optune callbacks
            if self.optimize_hyperparams:
                # Optune prune mechanism
                trial.report(valid_acc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return valid_acc

    @abstractclassmethod
    def calculate_acc(self, dataloader: DataLoader):
        pass

    def calculate_loss(self, dataloader: DataLoader):
        loss_sum = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                if self.gpu_boole:
                    x = x.cuda()
                    y = y.cuda()
                outputs = self.net(x)
                loss_sum += self.loss_metric(outputs, y) * y.shape[0]
                total += y.shape[0]

            loss = loss_sum / total
        return loss.cpu().data.item()

    def save_training_parameters(self, root: Path):
        train_params = {
            "lr": self.optimizer.param_groups[0]["lr"],
            "epochs": self.epochs,
            "batch": self.train_loader.batch_size,
            "opt": {"name": str(type(self.optimizer)), "parameters": self.optimizer.defaults},
        }

        with open(root / "trainer_parameters.json", "w") as f:
            json.dump(train_params, f, indent=3)

    def load_checkpoint(self, root: Path):
        self.checkpoint_handler, self.net, self.optimizer = CheckpointHandler.load_checkpoint_with_model(
            root, self.net, self.optimizer
        )
        self.init_epoch = self.checkpoint_handler.iteration + 1
        self.batch_count = self.checkpoint_handler.batch_count

        if self.checkpoint_handler.get_var("best_metric_to_opt"):
            self.best_metric_to_opt = self.checkpoint_handler.get_var("best_metric_to_opt")
            self.best_model_saver.best_metric = self.best_metric_to_opt

        # Update the checkpoint handler in all the cb
        metric_cal: EndOfEpochMetric
        for metric_cal in self.epoch_metrics_dict.values():
            metric_cal.checkpoint_handler = self.checkpoint_handler

    def save_checkpoint(self, filename):
        # Save training params json
        self.save_training_parameters(self.root)

        # Save Checkpoint
        self.checkpoint_handler.store_var(var_name="best_metric_to_opt", value=self.best_metric_to_opt)
        self.checkpoint_handler.store_var(var_name="last_epoch", value=self.final_epoch)
        self.checkpoint_handler.save_checkpoint(
            checkpoint_path=self.root / filename,
            iteration=self.final_epoch,
            batch_count=self.batch_count,
            model=self.net,
            optimizer=self.optimizer,
        )

    def __str__(self):
        train_params = {
            "lr": self.optimizer.param_groups[0]["lr"],
            "epochs": self.epochs,
            "batch": self.train_loader.batch_size,
            "opt": {"name": str(type(self.optimizer)), "parameters": self.optimizer.defaults},
        }
        return json.dumps(train_params, indent=3)
