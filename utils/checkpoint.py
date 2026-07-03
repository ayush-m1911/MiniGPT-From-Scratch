from pathlib import Path
import torch

class CheckpointManager:

    def __init__(self,checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_loss = float("inf")
    
    def save(self,model,optimizer,epoch,train_loss,val_loss,config):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        torch.save(
            checkpoint,
            self.checkpoint_dir / "latest.pt"
        )

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(
                checkpoint,
                self.checkpoint_dir / "best.pt"
            )
            return True
        return False

    def load(self,model,optimizer,path,device):
        checkpoint = torch.load(path, map_location=device)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        return checkpoint
        
        