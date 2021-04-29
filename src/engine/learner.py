import torch
import pytorch_lightning as pl
from engine.dispatcher import get_scheduler, get_criterion, get_metrics


class BCSNet(pl.LightningModule):
    def __init__(self, net, config=None):
        super().__init__()
        self.net = net
        self.config = config
        self.criterion = get_criterion(config)
        self._set_metrics(config)

    def forward(self, inputs):
        return self.net(inputs)

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.net.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params,
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
        images, targets = batch
        preds = self.net(images)
        loss = self.criterion(preds, targets)

        self.log(f"{mode}_loss", loss, prog_bar=False)

        if mode == "val":
            for metric_name in self.config["learner"]["metrics"]:
                Metric = self.__getattr__(f"{mode}_{metric_name}", None)
                if Metric is not None:
                    self.log(
                        f"{mode}_{metric_name}",
                        Metric(preds.argmax(axis=-1), targets),
                        prog_bar=True,
                    )
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode="val")

    def _set_metrics(self, config):
        """Set Pytorch Lightning Metrics as attributes."""
        for metric_name in config["learner"]["metrics"]:
            self.__setattr__(f"train_{metric_name}", get_metrics(metric_name, config))
            self.__setattr__(f"valid_{metric_name}", get_metrics(metric_name, config))
