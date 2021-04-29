from torch import nn
from torch.optim import lr_scheduler
import torchmetrics


def get_scheduler(optimizer, config):
    scheduler_config = config["learner"]["scheduler"]
    scheduler_type = scheduler_config["type"]

    if scheduler_type == "reduce_lr_on_plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_config["args_reduce_lr_on_plateau"]
        )

    # TODO: need to work on this (calculate the total number of steps)
    elif scheduler_type == "one_cycle":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            **scheduler_config["arg_one_cycle"],
            epochs=config["trainer"]["epochs"],
            steps_per_epoch=100
        )

    else:
        raise NotImplementedError("This scheduler is not implemented.")

    return scheduler


def get_criterion(config):
    if config["learner"]["criterion"] == "cross_entropy":
        return nn.CrossEntropyLoss()

    elif config["learner"]["criterion"] == "L1":
        return nn.L1Loss()

    else:
        raise NotImplementedError("This loss function is not implemented.")


def get_metrics(metric_name, config):
    if metric_name == "psnr":
        return torchmetrics.PSNR(data_range=1.0, dim=(-2, -1))

    elif metric_name == "ssim":
        return torchmetrics.SSIM(data_range=1.0)

    else:
        raise NotImplementedError("This metric is not implemented.")
