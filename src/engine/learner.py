import torch
import pytorch_lightning as pl
from model.bcsunet import BCSUNet
from engine.dispatcher import get_scheduler, get_criterion, get_metrics


class BCSUNetLearner(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.net = BCSUNet(config)
        self.config = config
        self.criterion = get_criterion(config)
        self._set_metrics(config)
        self.save_hyperparameters(config)

    def forward(self, inputs):
        reconstructed_image = self.net(inputs)
        return reconstructed_image

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["learner"]["lr"],
            weight_decay=self.config["learner"]["weight_decay"],
        )
        scheduler = get_scheduler(optimizer, self.config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def step(self, batch, mode="train"):
        inputs, targets = batch
        preds = self.net(inputs)
        loss = self.criterion(preds, targets)
        self.log(f"{mode}_loss", loss, prog_bar=True)

        if mode == "val":
            preds_ = preds.float().detach()
            for metric_name in self.config["learner"]["val_metrics"]:
                MetricClass = self.__getattr__(f"{mode}_{metric_name}")
                if MetricClass is not None:
                    self.log(
                        f"{mode}_{metric_name}",
                        MetricClass(preds_, targets),
                        prog_bar=True,
                    )
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode="val")

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self.net(inputs)
        for metric_name in self.config["learner"]["test_metrics"]:
            metric = self.__getattr__(f"test_{metric_name}")
            self.log(
                f"test_{metric_name}",
                metric(preds.float(), targets),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

    def _set_metrics(self, config):
        """
        Set TorchMetrics as attributes in a dynamical manner.
        For instance, `self.train_accuracy = torchmetrics.Accuracy()`
        """
        for metric_name in config["learner"]["val_metrics"]:
            # self.__setattr__(f"train_{metric_name}", get_metrics(metric_name, config))
            self.__setattr__(f"val_{metric_name}", get_metrics(metric_name, config))

        for metric_name in config["learner"]["test_metrics"]:
            self.__setattr__(f"test_{metric_name}", get_metrics(metric_name, config))
