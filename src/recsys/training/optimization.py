"""Optimizer setup and gradient updates for a neural training epoch."""

import numpy as np
import torch


def make_optimizer(model, options):
    """Create the optimizer, scheduler, and gradient scaler for one fit."""
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}.get(
        options["optimizer"]
    )
    if optimizer_class is None:
        raise ValueError("optimizer must be adam or adamw")
    optimizer = optimizer_class(
        model.parameters(),
        lr=options["learning_rate"],
        weight_decay=options["weight_decay"],
    )
    schedule = options["scheduler"]
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=schedule["step_size"], gamma=schedule["gamma"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=options["amp"])
    return optimizer, scheduler, scaler


def optimize_epoch(model, batches, optimizer, scheduler, scaler, options, device):
    """Update model parameters and return the mean batch loss and number of updates."""
    losses = []
    for step, batch in enumerate(batches):
        if options["max_batches"] is not None and step >= options["max_batches"]:
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            enabled=options["amp"],
        ):
            loss = model.loss(*batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    if not losses:
        raise ValueError(
            "No trainable examples (check target split, history and negative candidates)"
        )
    scheduler.step()
    return float(np.mean(losses)), len(losses)
