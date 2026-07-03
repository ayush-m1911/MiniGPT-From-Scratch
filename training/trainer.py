import torch 
import torch.nn as nn
from tqdm import tqdm

class Trainer:

    def __init__(self,model,train_loader,config,optimizer,device,logger):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.criterion = nn.CrossEntropyLoss()
        self.model.to(device)
    
    def train_step(self,x,y):
        x = x.to(self.device)
        y = y.to(self.device)
        logits, _ = self.model(x)
        loss = self.criterion(
            logits.view(-1,logits.size(-1)),
            y.view(-1)
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress = tqdm(self.train_loader)

        for x,y in progress:
            loss = self.train_step(x,y)

            total_loss+=loss

            progress.set_postfix(
                loss=f"{loss:.4f}"
            )
        return total_loss / len(self.train_loader)

    def train(self):
        self.logger.info("Training Started...")

        for epoch in range(
            self.config.epochs
        ):
          loss = self.train_epoch()
          self.logger.info(
            f"Epoch {epoch+1}:"
            f"Loss = {loss:.4f}"
          )
        self.logger.info(
            "Training Finished"
        )