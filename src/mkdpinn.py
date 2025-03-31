import numpy as np
import torch.nn as nn
import torch
from torch.optim import Adam
from src.utils.early_stopping import EarlyStopping
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm
import warnings
from src.models.components import HSM,PREDICTOR,PGR


warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


log_file = open("training.log", "a")


def write_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"{timestamp} - {message}\n")
    log_file.flush()

class MKDPINN(nn.Module):
    def __init__(self, hidden_dim, derivatives_order, meta_params):
        super(MKDPINN, self).__init__()
        self.hidden_dim = hidden_dim
        self.order = derivatives_order
        self.input_dim = 1 + self.hidden_dim * (self.order + 1)
        self.lr = meta_params[0]
        self.inner_steps = meta_params[1]
        self.outer_step_size = meta_params[2]
        self.outer_iterations = meta_params[3]
        self.inner_batch_size = meta_params[4]
        self.meta_batch_size = meta_params[5]
        self.n_shot = meta_params[6]

        self.hsm = HSM(output_dim=self.hidden_dim).to(device)
        self.predictor = PREDICTOR(self.hidden_dim + 1).to(device)
        self.pgr = PGR(self.input_dim, 1).to(device)
        self.optim = Adam(params=[{'params': self.hsm.parameters()},
                                  {'params': self.predictor.parameters()},
                                  {'params': self.pgr.parameters()}],
                          lr=self.lr)

    # scoring function
    def Score(self, pred, true):
        score = 0
        for i in range(pred.shape[0]):
            h = pred[i] - true[i]
            if h >= 0:
                s = torch.exp(h / 10) - 1
            else:
                s = torch.exp(-h / 13) - 1
            score += s
        return score

    # predict RUL and hidden state
    def predict_rul_and_hidden(self, x, t):
        hidden = self.hsm(x)
        hidden.requires_grad_(True)
        return self.predictor(torch.concat([hidden, t], dim=1)), hidden

    # Find the partial derivatives
    def compute_pde_residual(self, x, t):
        t.requires_grad_(True)
        u, h = self.predict_rul_and_hidden(x, t)
        u = u.reshape(-1, 1)
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_h = [u]
        for i in range(self.order):
            u_ = torch.autograd.grad(
                u_h[-1], h,
                grad_outputs=torch.ones_like(u_h[-1]),
                retain_graph=True,
                create_graph=True
            )[0]
            u_h.append(u_)
        deri = h
        for data in u_h:
            deri = torch.concat([deri, data], dim=1)
        f = u_t - self.pgr(deri)
        return f

    def train_model(self, dataset, dataset_val, dataset_test, resume_from_checkpoint=False):
        MSE = nn.MSELoss()

        if self.n_shot != 0:
            print(
                "\n Warning: You have set n_shot to a value other than 0, but the provided dataset is designed for 0-shot validation only "
                "(the validation set likely belongs to the same task as the training set). \n"
                "To ensure proper evaluation on this dataset, the model will still perform 0-shot validation. "
                "For few-shot learning experiments, please use a dataset with distinct tasks for training and validation.\n"
                f"Current n_shot value: {self.n_shot}, but running in 0-shot mode."
            )

        early_stopping = EarlyStopping(patience=250, path1='./Result/hsm.pth', path2='./Result/predictor.pth',
                                       path3='./Result/pgr.pth')

        write_log("Starting model training...")

        if resume_from_checkpoint:
            print("Resuming from checkpoint...")
            write_log("Resuming from checkpoint...")
            early_stopping.load_checkpoint(self.hsm, self.predictor, self.pgr)

        print(
            "The full training code is currently withheld for peer review. "
            "It will be released upon paper acceptance. "
            "For now, please use the trained weights (`model.test()`). "
            "Contact wy1475899830@163.com for exceptions."
        )

        for epoch in range(self.outer_iterations):
            write_log(f"Starting epoch {epoch + 1}/{self.outer_iterations}")
            task_loss_total = 0
            task_count = 0
            self.train()

            dataset_list = list(dataset)
            total_tasks = len(dataset_list)



        log_file.close()
    def validate(self, validation_dataset, epoch, MSE, early_stopping):
        self.eval()
        valid_loss = 0
        for x_valid, t_valid, rul_valid in validation_dataset:
            t_valid = t_valid.to(device).reshape(-1, 1)
            x_valid = x_valid.to(device)
            rul_valid = rul_valid.to(device)
            rul_valid_pred, _ = self.predict_rul_and_hidden(x_valid, t_valid)
            valid_loss += MSE(rul_valid_pred, rul_valid)

        valid_loss /= len(validation_dataset)  # Calculate the average validation loss
        early_stopping(valid_loss, model1=self.hsm, model2=self.predictor, model3=self.pgr)

        if epoch % 1 == 0:
            write_log(f'Epoch: {epoch + 1},   Valid_RUL_RMSE: {torch.sqrt(torch.tensor(valid_loss)):.4f}')
            print(f'Epoch: {epoch + 1},   Valid_RUL_RMSE: {torch.sqrt(torch.tensor(valid_loss)):.4f}')
    def test(self, dataset_test, MSE, in_train=True,dataset=False):

        self.eval()
        ruls = []
        ruls_t = []
        # Feed all data into the model at once
        x_valid_all = []
        t_valid_all = []
        rul_valid_all = []

        # Collect all samples
        for x_valid, t_valid, rul_valid in dataset_test:
            t_valid_all.append(t_valid.to(device).reshape(-1, 1))
            x_valid_all.append(x_valid.to(device))
            rul_valid_all.append(rul_valid.to(device))

        # Convert list to tensor
        x_valid_all = torch.cat(x_valid_all)
        t_valid_all = torch.cat(t_valid_all)
        rul_valid_all = torch.cat(rul_valid_all)

        if not in_train:
            if not dataset:
                self.hsm.load_state_dict(torch.load('./Result/hsm.pth'))
                self.predictor.load_state_dict(torch.load('./Result/predictor.pth'))
            else:
                self.hsm.load_state_dict(torch.load(f'./Result/{dataset}/hsm.pth'))
                self.predictor.load_state_dict(torch.load(f'./Result/{dataset}/predictor.pth'))

        # Use the model to make predictions
        rul_valid_pred, _ = self.predict_rul_and_hidden(x_valid_all, t_valid_all)

        # Calculate loss
        valid_loss = MSE(rul_valid_pred, rul_valid_all)

        # Get the prediction and actual value of the last time step
        ruls.append(rul_valid_pred[-1].detach().cpu().numpy())
        ruls_t.append(rul_valid_all[-1].detach().cpu().numpy())

        # Calculate the score
        score = self.Score(rul_valid_pred.detach().cpu(), rul_valid_all.detach().cpu())

        if in_train:
            write_log(f'Test RMSE: {torch.sqrt(valid_loss):.4f}, Score: {score:.4f} on zero-shot mode')

        print(f'Test RMSE: {torch.sqrt(valid_loss):.4f}, Score: {score:.4f} on zero-shot mode')
