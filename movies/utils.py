from typing import Any, Callable, Mapping, Optional, Union
from lightning import LightningModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MetricCollection

class BaseModel(LightningModule):
    """Lightning boilerplate code for logging, accumulating and logging metrics every epoch.
    """

    def __init__(self, *args, learning_rate=1e-4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train_metrics : MetricCollection = None
        self.val_metrics : MetricCollection = None
        self.test_metrics : MetricCollection = None 
        self.learning_rate = learning_rate
    
    def get_metrics_for_subset(self, subset):
        if subset == "train":
            return self.train_metrics
        elif subset == "val":
            return self.val_metrics
        else:
            return self.test_metrics

    def update_metrics(self, pred, y, subset="train"):
        metrics = self.get_metrics_for_subset(subset)
        if metrics is not None:
            metrics.update(pred, y)

    def get_metrics(self, subset="train"):
        metrics = self.get_metrics_for_subset(subset)
        if metrics is not None:
            v = metrics.compute()
            res = {}
            for metric_name, metric_value in v.items():
                res[f'{subset}_{metric_name}'] = metric_value.item()
            return res

    def reset_metrics(self, subset="train"):
        metrics = self.get_metrics_for_subset(subset)
        if metrics is not None:
            for metric in metrics.values():
                metric.reset()

    def on_train_start(self) -> None:
        if self.train_metrics:
            for metric in self.train_metrics.values():
                metric.to(self.device)

    def on_validation_start(self) -> None:
        if self.val_metrics:
            for metric in self.val_metrics.values():
                metric.to(self.device)

    def on_test_start(self) -> None:
        if self.test_metrics:
            for metric in self.test_metrics.values():
                metric.to(self.device)
    
    def on_train_epoch_end(self) -> None:
        self.log_dict(self.get_metrics())
        self.reset_metrics()
    
    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.get_metrics(subset="val"))
        self.reset_metrics(subset="val")
    
    def on_test_epoch_end(self) -> None:
        self.log_dict(self.get_metrics(subset="test"))
        self.reset_metrics(subset="test")
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def common_step(self, batch):
        raise NotImplementedError

    def training_step(self, batch, _) -> STEP_OUTPUT:
        labels = batch['rating']
        y_hat = self.common_step(batch)
        self.update_metrics(y_hat, labels, subset="train")
        loss = self.loss(y_hat.float(), labels.float())
        return loss
    
    def validation_step(self, batch, _) -> STEP_OUTPUT | None:
        labels = batch['rating']
        y_hat = self.common_step(batch)
        self.update_metrics(y_hat, labels, subset="val")
        return self.loss(y_hat.float(), labels.float())
    
    def forward(self, batch) -> Any:
        return self.common_step(batch)
    
    def log_dict(self, dictionary, *args, **kwargs) -> None:
        if dictionary is None:
            dictionary = {}
        return super().log_dict(dictionary=dictionary, *args, **kwargs)