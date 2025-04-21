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

        for epoch in range(self.outer_iterations):
            write_log(f"Starting epoch {epoch + 1}/{self.outer_iterations}")
            task_loss_total = 0
            task_count = 0
            self.train()

            dataset_list = list(dataset)
            total_tasks = len(dataset_list)


            with tqdm(total=total_tasks, desc=f"Epoch {epoch + 1}/{self.outer_iterations}", unit=" meta-tasks") as pbar:
                for i in range(0, total_tasks, self.meta_batch_size):

                    meta_batch = dataset_list[i:min(i + self.meta_batch_size, total_tasks)]

                    # Save the original model parameters
                    original_weights_xnn = deepcopy(self.hsm.state_dict())
                    original_weights_mlp = deepcopy(self.predictor.state_dict())
                    original_weights_deepHPM = deepcopy(self.pgr.state_dict())

                    # Store updated parameters for all tasks
                    updated_weights = {
                        'hsm': [],
                        'predictor': [],
                        'pgr': []
                    }

                    batch_loss = 0
                    for task_idx, meta_task_data in enumerate(meta_batch):
                        x_task, t_task, rul_task = meta_task_data
                        num_samples = x_task.size(0)
                        x_task = x_task.to(device)
                        rul_task = rul_task.to(device)
                        t_task = t_task.to(device)

                        # Reset the model to its original parameters
                        self.hsm.load_state_dict(original_weights_xnn)
                        self.predictor.load_state_dict(original_weights_mlp)
                        self.pgr.load_state_dict(original_weights_deepHPM)

                        # Create a task-specific optimizer
                        task_optim = Adam([
                            {'params': self.hsm.parameters()},
                            {'params': self.predictor.parameters()},
                            {'params': self.pgr.parameters()}
                        ], lr=self.lr)

                        # Inner loop training
                        inner_batch_size = min(self.inner_batch_size, num_samples)
                        for inner_step in range(self.inner_steps):
                            indices = torch.randperm(num_samples).to(device)
                            for start_idx in range(0, num_samples, inner_batch_size):
                                batch_indices = indices[start_idx:start_idx + inner_batch_size]
                                x_batch = x_task[batch_indices]
                                rul_batch = rul_task[batch_indices]
                                t_batch = t_task[batch_indices].reshape(-1, 1)

                                # Forward propagation and loss calculation
                                u, h = self.predict_rul_and_hidden(x_batch, t_batch)
                                f = self.compute_pde_residual(x_batch, t_batch)
                                loss1 = MSE(u, rul_batch)
                                loss2 = MSE(f, torch.zeros(f.shape).to(device))
                                loss = loss1 + loss2

                                # Inner loop optimization - using task-specific optimizer
                                task_optim.zero_grad()
                                loss.backward()
                                task_optim.step()

                        # Record the final loss of this task
                        batch_loss += loss.item()

                        # Store updated parameters
                        updated_weights['hsm'].append(deepcopy(self.hsm.state_dict()))
                        updated_weights['predictor'].append(deepcopy(self.predictor.state_dict()))
                        updated_weights['pgr'].append(deepcopy(self.pgr.state_dict()))

                        task_count += 1
                        pbar.update(1)

                    # Calculate the average loss
                    batch_size = len(meta_batch)
                    batch_loss /= batch_size
                    task_loss_total += batch_loss

                    # meta-update: restore original parameters and move in the updated direction
                    self.hsm.load_state_dict(original_weights_xnn)
                    self.predictor.load_state_dict(original_weights_mlp)
                    self.pgr.load_state_dict(original_weights_deepHPM)

                    # Perform a meta-update for each model - make sure the update is in the right direction
                    for model_name, model in zip(['hsm', 'predictor', 'pgr'], [self.hsm, self.predictor, self.pgr]):
                        original_weights = original_weights_xnn if model_name == 'hsm' else (
                            original_weights_mlp if model_name == 'predictor' else original_weights_deepHPM)

                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                # Calculate the average of the parameter differences
                                update = torch.zeros_like(param.data)
                                for task_weights in updated_weights[model_name]:
                                    update += task_weights[name] - original_weights[name]
                                update /= batch_size

                                # Applying meta updates
                                param.data = original_weights[name] + self.outer_step_size * update

                    # Update progress bar information
                    pbar.set_postfix(avg_loss=task_loss_total / task_count)

                # Verify and test after each epoch
                write_log(f"Epoch {epoch + 1} completed. Average task loss: {task_loss_total / task_count:.4f}")
                self.validate(dataset_val, epoch, MSE, early_stopping)
                self.test(dataset_test, MSE)

                if early_stopping.early_stop:
                    write_log("Early stopping triggered.")
                    print("Early Stopping")
                    log_file.close()
                    break

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
