import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import joblib
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import joblib

class HaralickClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.001):
        super(HaralickClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.learning_rate = learning_rate
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.max_val_acc = 0.0

    def forward(self, x):
        x = F.relu(self.fc1(x))  
        # x = F.relu(self.fc2(x))  
        x = self.output_layer(x)  # Output layer WITHOUT activation
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        y_hat = F.softmax(y_hat, dim=1)
        loss = self.loss_fn(y_hat.log(), F.one_hot(y, num_classes=2).float())
        preds = torch.argmax(y_hat, dim=1)
        acc = balanced_accuracy_score(y.cpu(), preds.cpu())
        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, logger=True, on_epoch=True)
        return loss
        

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        # print(f'\ny shape: {y.shape}, y_hat shape: {y_hat.shape}')
        y_hat = F.softmax(y_hat, dim=1)
        loss = self.loss_fn(y_hat.log(), F.one_hot(y, num_classes=2).float())
        preds = torch.argmax(y_hat, dim=1)
        acc = balanced_accuracy_score(y.cpu(), preds.cpu())
        self.max_val_acc = max(self.max_val_acc, acc)  # Update max validation accuracy
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=True)
        self.log('val_acc', acc, prog_bar=True, logger=True, on_step=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
    
    def reset_max_val_acc(self):
        self.max_val_acc = 0.0

class SimpleMLPWrapper:
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=(2, 2), learning_rate=0.001, batch_size=32, exp_name='default'):
        self.model = HaralickClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, learning_rate=learning_rate)
        self.batch_size = batch_size
        self.exp_name = exp_name

    def fit(self, X_train, y_train, X_val=None, y_val=None, num_epochs=10, log_name="my_model"):
        self.model.train()

        X_train = np.stack(X_train, axis=0)
        y_train = np.stack(y_train, axis=0)
        if X_val is not None and y_val is not None:
            X_val = np.stack(X_val, axis=0)
            y_val = np.stack(y_val, axis=0)

        # Scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
        if X_val is not None:
            X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)

        # Create datasets
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        else:
            val_dataloader = None

        logger = TensorBoardLogger(f"tb_logs/{self.exp_name}", name=log_name)

        trainer = pl.Trainer(max_epochs=num_epochs, accelerator='gpu', logger=logger)
        trainer.fit(self.model, train_dataloader, val_dataloader)
    
    def predict(self, X_test):
        self.model.eval()
        scaler = StandardScaler()


        X_test = np.stack(X_test, axis=0)
        X_test = scaler.fit_transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
        
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size)

        preds = []
        for batch in test_dataloader:
            x = batch[0]
            y_hat = F.softmax(self.model(x).squeeze(1), dim=1)
            # print(f'y_hat shape: {y_hat.shape}')
            preds.append(torch.argmax(y_hat, dim=1).cpu().numpy())
        return np.concatenate(preds)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_test = np.stack(y_test, axis=0)
        return balanced_accuracy_score(y_test, y_pred)
    
    def save_model(self, filepath):
        joblib.dump(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        state_dict = joblib.load(filepath)
        self.model.load_state_dict(state_dict)
