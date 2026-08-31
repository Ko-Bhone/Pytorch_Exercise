"""
PROJECT 1 — House Price Prediction
Regression | MSELoss | preprocessing | DataLoader | EarlyStopping
Clean OOP / end-to-end training pipeline
"""

from dataclasses import dataclass
from pathlib import Path
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

@dataclass
class Config:
    data_path: str = "Dataset/house_price.csv"
    target_column: str = "SalePrice"
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 1000          # upper safety limit, not a manually chosen final epoch
    patience: int = 30
    min_delta: float = 1e-5
    random_seed: int = 42
    checkpoint_path: str = "checkpoints/house_price_best.pt"


class Utils:
    @staticmethod
    def seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


class HousePriceDataModule:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = None
        self.feature_count = None
        self.test_raw = None

    def _build_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
        categorical = X_train.select_dtypes(exclude=["number"]).columns.tolist()

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

        return ColumnTransformer([
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical)])

    def setup(self):
        path = Path(self.config.data_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {path}\n"
                "Download Kaggle House Prices train.csv and place it there.")
        df = pd.read_csv(path, sep="\t")
        df = df.drop(columns=["Id"], errors="ignore")
        if self.config.target_column not in df.columns:
            raise ValueError(
                f"Target '{self.config.target_column}' not found. "
                f"Columns: {df.columns.tolist()}")
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column].astype(np.float32)
        # Log target makes the price distribution easier for a neural network.
        y = np.log1p(y)
        X_train, X_temp, y_train, y_temp = train_test_split(

            X, y, test_size=0.30, random_state=self.config.random_seed)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=self.config.random_seed)
        print("\nTARGET DISTRIBUTION")
        print("Train:")
        print(y_train.describe())


        print("\nValidation:")
        print(y_val.describe())
        print("\nTest:")
        print(y_test.describe())
        self.preprocessor = self._build_preprocessor(X_train)
        X_train = self.preprocessor.fit_transform(X_train).astype(np.float32)
        X_val = self.preprocessor.transform(X_val).astype(np.float32)
        X_test = self.preprocessor.transform(X_test).astype(np.float32)
        self.feature_count = X_train.shape[1]
        train_ds = TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train.to_numpy(np.float32)).view(-1, 1))
        val_ds = TensorDataset(
            torch.tensor(X_val),
            torch.tensor(y_val.to_numpy(np.float32)).view(-1, 1))
        test_ds = TensorDataset(
            torch.tensor(X_test),
            torch.tensor(y_test.to_numpy(np.float32)).view(-1, 1))
        return (
            DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=256, shuffle=False),
            DataLoader(test_ds, batch_size=256, shuffle=False))

class HousePriceModel(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            # nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1))

    def forward(self, x):
        return self.network(x)

class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total_items = 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(X)
            loss = self.criterion(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total_loss += loss.item() * X.size(0)
            total_items += X.size(0)
        return total_loss / total_items

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            prediction = self.model(X)
            loss = self.criterion(prediction, y)
            total_loss += loss.item() * X.size(0)
            total_items += X.size(0)
        return total_loss / total_items

    def fit(self, train_loader, val_loader):
        stopper = EarlyStopping(
            self.config.patience,
            self.config.min_delta)

        Path(self.config.checkpoint_path).parent.mkdir(
            parents=True, exist_ok=True)

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.5f} | "
                f"val_loss={val_loss:.5f}")
            if stopper.step(val_loss, self.model):
                print("Early stopping triggered.")
                break
            torch.save({
                    "model_state_dict": stopper.best_state,
                    "best_val_loss": stopper.best_loss,
                    "epoch": epoch},
                self.config.checkpoint_path,)
        stopper.restore(self.model)
        print(f"Best validation loss: {stopper.best_loss:.5f}")
        return self.train_losses, self.val_losses

class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict_loader(self, loader):
        self.model.eval()
        predictions, targets = [], []
        for X, y in loader:
            prediction = self.model(X.to(self.device))
            predictions.append(prediction.cpu())
            targets.append(y)
        return (
            torch.cat(predictions).numpy().ravel(),
            torch.cat(targets).numpy().ravel())

    def evaluate(self, loader):
        pred_log, true_log = self.predict_loader(loader)
        pred = np.expm1(pred_log)
        true = np.expm1(true_log)
        print("\nLOG SPACE CHECK")
        print("Pred log min:", pred_log.min())
        print("Pred log max:", pred_log.max())
        print("True log min:", true_log.min())
        print("True log max:", true_log.max())
        print("\nPRICE RANGE")
        print("Pred price min:", pred.min())
        print("Pred price max:", pred.max())
        print("True price min:", true.min())
        print("True price max:", true.max())
        print("\nTEST RESULTS")
        print(f"RMSE: {mean_squared_error(true, pred) ** 0.5:,.2f}")
        print(f"MAE : {mean_absolute_error(true, pred):,.2f}")
        print(f"R2  : {r2_score(true, pred):.4f}")
        print("\nSAMPLE PREDICTION")
        for i in range(min(10, len(pred))):
            print(
                f"Actual: ${true[i]:,.2f} |"
                f"Predicted: ${pred[i]:,.2f}")
        return pred, true

class LossVisualizer:
    @staticmethod
    def plot(train_losses, val_losses):
        plt.figure(figsize=(10,6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

class PredictionVisualizer:
    @staticmethod
    def plot(true, pred):
        plt.figure(figsize=(10,6))
        plt.scatter(true, pred, alpha=0.6)
        min_value = min(true.min(), pred.min())
        max_value = max(true.min(), pred.min())
        plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--", label="Perfect Prediction")
        plt.xlabel("Actual House Price")
        plt.ylabel("Predicted House Price")
        plt.title("Actual vs Predicted House Price")
        plt.legend()
        plt.grid(True)
        plt.show()

class ResidualVisualizer:
    @staticmethod
    def plot(true, pred):
        residual = true - pred
        print("\nRESIDUAL ANALYSIS")
        print(f"Mean Residual: {residual.mean():,.2f}")
        print(f"Std Residual: {residual.std():,.2f}")
        print(f"Min Residual: {residual.min():,.2f}")
        print(f"Max Residual: {residual.max():,.2f}")
        plt.figure(figsize=(10,6))
        plt.scatter(pred, residual, alpha=0.6)
        plt.axhline(y=0, linestyle='--')
        plt.xlabel("Predicted House Price")
        plt.ylabel("Residual (Actual - Predicted)")
        plt.title("Residual Analysis")
        plt.grid(True)
        plt.show()

class HousePriceApp:
    def __init__(self):
        self.config = Config()
        Utils.seed_everything(self.config.random_seed)
        self.device = Utils.get_device()
        self.data = HousePriceDataModule(self.config)
        self.model = None
        self.trainer = None

    def run(self):
        print("Device:", self.device)
        train_loader, val_loader, test_loader = self.data.setup()
        self.model = HousePriceModel(self.data.feature_count).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8,)
        self.trainer = Trainer(
            self.model, criterion, optimizer,
            scheduler, self.device, self.config)
        train_losses, val_losses = self.trainer.fit(train_loader, val_loader)
        LossVisualizer.plot(train_losses, val_losses)
        evaluator = Evaluator(self.model, self.device)
        pred, true = evaluator.evaluate(test_loader)
        PredictionVisualizer.plot(true, pred)
        ResidualVisualizer.plot(true, pred)



if __name__ == "__main__":
    HousePriceApp().run()