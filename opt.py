import torch
from loss import multi_bce
from metrics import compute_metrics
from utils import save_eval_images, save_sample_images, save_features_per_channel
from logger import MetricLogger, SmoothedValue


def train_one_epoch(
    args, model, data_loader, optimizer, epoch, print_freq=10, log_file=""
):
    """Train for one epoch"""
    model.train()

    metric_logger = MetricLogger(delimiter="  ", log_file=log_file)

    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Train epoch: [{epoch}]"

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        inputs = batch.get("images", batch.get("inputs")).to(args.device)
        targets = batch.get("masks", batch.get("targets")).to(args.device)
        outputs = model(inputs)
        total_loss, loss_m = multi_bce(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update learning rate for logging
        for param_group in optimizer.param_groups:
            metric_logger.update(lr=param_group["lr"])

        metric_logger.update(**{"loss_m": loss_m.item()})
        metric_logger.update(**{"loss_total": total_loss.item()})
        # Save sample images
        if batch_idx % (print_freq * 5) == 0 and hasattr(args, "output_dir"):
            save_sample_images(
                inputs, outputs, targets, batch_idx, epoch, args.output_dir
            )

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Train stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_fn(
    args,
    data_loader,
    model,
    epoch,
    print_freq=100,
    log_file="",
):
    """Evaluate forgery detection model"""
    model.eval()

    metric_logger = MetricLogger(delimiter="  ", log_file=log_file)
    header = f"Test: [{epoch}]"

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            inputs = batch.get("images", batch.get("inputs")).to(args.device)
            targets = batch.get("masks", batch.get("targets")).to(args.device)
            filenames = batch.get(
                "filenames",
                [f"sample_{batch_idx}_{i}" for i in range(inputs.size(0))],
            )

            outputs = model(inputs)
            pred_masks = outputs[2]
            total_loss, loss_m = multi_bce(outputs, targets)

            metric_logger.update(**{"loss_m": loss_m.item()})
            metric_logger.update(**{"loss_total": total_loss.item()})

            metrics = compute_metrics(pred_masks, targets, threshold=0.5)

            for metric_name, metric_value in metrics.items():
                metric_logger.update(**{f"{metric_name}": metric_value})

            save_eval_images(inputs, pred_masks, targets, filenames, args.output_dir)

    metric_logger.synchronize_between_processes()
    print(f"Test stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
