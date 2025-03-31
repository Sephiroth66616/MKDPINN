import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=50, verbose=True, path1='./Result/hsm.pth', path2='./Result/predictor.pth', path3='./Result/pgr.pth'):
        self.patience = patience
        self.verbose = verbose
        self.path1 = path1
        self.path2 = path2
        self.path3 = path3
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model1, model2, model3):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model1, model2, model3)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model1, model2, model3)
            self.counter = 0

    def save_checkpoint(self, val_loss, model1, model2, model3):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model1.state_dict(), self.path1)
        torch.save(model2.state_dict(), self.path2)
        torch.save(model3.state_dict(), self.path3)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model1, model2, model3):
        model1.load_state_dict(torch.load(self.path1))
        model2.load_state_dict(torch.load(self.path2))
        model3.load_state_dict(torch.load(self.path3))
