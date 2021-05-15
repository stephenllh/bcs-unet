import torch
import pytorch_lightning as pl
from .net import ReconNet
from engine.dispatcher import get_scheduler, get_criterion, get_metrics


class ReconNetLearner(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        y_dim = int(config["sampling_ratio"] * config["img_dim"] ** 2)
        self.net = ReconNet(y_dim, config["img_dim"])
        self.config = config
        self.criterion = get_criterion(config)
        self._set_metrics(config)
        self.save_hyperparameters(config)

    def forward(self, inputs):
        return self.net(inputs)

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.net.parameters())
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.config["learner"]["lr"],
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

        # preds_ = preds.float().detach().cpu()
        # targets_ = targets.detach().cpu()

        # # Log validation metrics
        # if mode == "val":
        #     for metric_name in self.config["learner"]["metrics"]:
        #         metric = self.__getattr__(f"{mode}_{metric_name}")
        #         self.log(
        #             f"{mode}_{metric_name}",
        #             metric(preds_, targets_),
        #             prog_bar=True,
        #         )
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
                metric(preds, targets),
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
