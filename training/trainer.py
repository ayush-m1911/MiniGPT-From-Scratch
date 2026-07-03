import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """
    Handles the complete training pipeline for the GPT model.
    """

    def __init__(
        self,
        model,
        train_loader,
        config,
        optimizer,
        device,
        logger
    ):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.optimizer = optimizer
        self.device = device
        self.logger = logger

        self.criterion = nn.CrossEntropyLoss()

        self.model.to(self.device)

    def forward_pass(self, x, y):
        """
        Performs only the forward pass and computes the loss.
        """

        x = x.to(self.device)
        y = y.to(self.device)

        logits, _ = self.model(x)

        loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        return loss

    def optimization_step(self, loss):
        """
        Performs backward propagation and updates model weights.
        """

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

    def train_step(self, x, y):
        """
        Executes one complete training step.
        """

        loss = self.forward_pass(x, y)

        self.optimization_step(loss)

        return loss.item()

    def train_epoch(self, epoch):
        """
        Trains the model for one epoch.
        """

        self.model.train()

        total_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}"
        )

        for x, y in progress:

            loss = self.train_step(x, y)

            total_loss += loss

            progress.set_postfix(
                loss=f"{loss:.4f}"
            )

        average_loss = total_loss / len(self.train_loader)

        return average_loss

    def after_epoch(self, epoch, train_loss):
        """
        Operations to perform after every epoch.

        Future additions:
        - Validation
        - Scheduler Step
        - Save Checkpoint
        - Generate Sample Text
        """

        self.logger.info("-" * 60)

        self.logger.info(
            f"Epoch {epoch + 1}/{self.config.epochs}"
        )

        self.logger.info(
            f"Training Loss : {train_loss:.4f}"
        )

        self.logger.info("-" * 60)

    def train(self):
        """
        Complete training loop.
        """

        self.logger.info("=" * 60)
        self.logger.info("Training Started")
        self.logger.info("=" * 60)

        for epoch in range(self.config.epochs):

            train_loss = self.train_epoch(epoch)

            self.after_epoch(
                epoch,
                train_loss
            )

        self.logger.info("=" * 60)
        self.logger.info("Training Finished")
        self.logger.info("=" * 60)