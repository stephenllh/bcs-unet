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
        _, reconstructed_image = self.net(inputs)
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
        preds1, preds2 = self.net(inputs)
        if self.config["learner"]["intermediate_image"]:
            loss1 = self.criterion(preds1, targets)
            loss2 = self.criterion(preds2, targets)
            loss = loss1 + loss2
        else:
            loss = self.criterion(preds2, targets)
        self.log(f"{mode}_loss", loss, prog_bar=False)

        if mode == "val":
            preds_ = preds2.float().detach()
            for metric_name in self.config["learner"]["metrics"]:
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
        return self.step(batch, mode="val")

    def _set_metrics(self, config):
        """Set Pytorch Lightning Metrics as attributes."""
        for metric_name in config["learner"]["metrics"]:
            self.__setattr__(f"train_{metric_name}", get_metrics(metric_name, config))
            self.__setattr__(f"val_{metric_name}", get_metrics(metric_name, config))
