# ============================================================
# Project 02 - Iris Classification
# PyTorch + OOP
# Intermediate Level
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

# CONFIG
class Config:
    CSV_PATH = "Dataset/iris.csv"
    MODEL_PATH = "iris_model.pth"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    BATCH_SIZE = 16
    LEARNING_RATE = 0.01
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
        else "cpu")
    FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    LABEL_MAPPING = {"setosa": 0, "versicolor": 1, "virginica": 2}
    REVERSE_MAPPING = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"}

# SET RANDOM SEED
class SeedManager:
    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

# DATA LOADER
class IrisDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset not found: {self.file_path}")
        df = pd.read_csv(self.file_path, sep=None, engine="python")
        # Fix literal \t in column names/data if present
        if len(df.columns) == 1 and "\\t" in df.columns[0]:
            df = pd.read_csv(self.file_path, sep=r"\\t", engine="python")
        return df

    def show_info(self, df):
        print("\n" + "=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nShape:")
        print(df.shape)
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("\nClass Distribution:")
        print(df["species"].value_counts())

# DATA PREPROCESSOR
class IrisPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
    def prepare_data(self, df):
        # Features
        x = df[Config.FEATURES].values
        # Target
        y = df["species"].map(Config.LABEL_MAPPING).values
        # Train / Test Split
        x_train, x_test, y_train, y_test = (train_test_split(x, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y))
        # Scaling
        x_train = self.scaler.fit_transform(x_train)
        x_test = self.scaler.transform(x_test)
        return (x_train, x_test, y_train, y_test)

# PYTORCH DATASET
class IrisDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

# NEURAL NETWORK
class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # Input Layer
            nn.Linear(4, 16),
            nn.ReLU(),
            # Hidden Layer
            nn.Linear(16, 32),
            nn.ReLU(),
            # Hidden Layer
            nn.Linear(32, 16),
            nn.ReLU(),
            # Output Layer
            nn.Linear(16, 3))

    def forward(self, x):
        return self.network(x)

# TRAINER
class IrisTrainer:
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.criterion = (nn.CrossEntropyLoss())
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE)
        self.loss_history = []
        self.accuracy_history = []

    def train(self):
        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)
        for epoch in range(Config.EPOCHS):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            # Mini Batch Training
            for x_batch, y_batch in (self.train_loader):
                x_batch = x_batch.to(Config.DEVICE)
                y_batch = y_batch.to(Config.DEVICE)
                # Forward Pass
                outputs = self.model(x_batch)
                # Loss
                loss = self.criterion(outputs, y_batch)
                # Clear Gradient
                self.optimizer.zero_grad()
                # Backpropagation
                loss.backward()
                # Update Weights
                self.optimizer.step()
                total_loss += loss.item()
                # Prediction
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == y_batch).sum().item()
                total += (y_batch.size(0))
            epoch_loss = (total_loss / len(self.train_loader))
            epoch_accuracy = (correct / total)
            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_accuracy)
            if (
                (epoch + 1) % 10 == 0
                or epoch == 0):
                print(
                    f"Epoch "
                    f"{epoch + 1:03d}/"
                    f"{Config.EPOCHS} | "
                    f"Loss: "
                    f"{epoch_loss:.4f} | "
                    f"Accuracy: "
                    f"{epoch_accuracy:.4f}"
                )
        print("\nTraining completed!")

# EVALUATOR
class IrisEvaluator:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader

    def predict(self):
        self.model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for x_batch, y_batch in (self.test_loader):
                x_batch = x_batch.to(Config.DEVICE)
                outputs = self.model(x_batch)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        return (np.array(all_labels), np.array(all_predictions))

    def evaluate(self):
        y_true, y_pred = self.predict()
        accuracy = accuracy_score(y_true, y_pred)
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        print(
            f"\nTest Accuracy: "
            f"{accuracy * 100:.2f}%"
        )
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["setosa", "versicolor", "virginica"]))
        return (y_true, y_pred, accuracy)

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        display = (ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["setosa", "versicolor", "virginica"]))
        display.plot()
        plt.title("Iris Classification - Confusion Matrix")
        plt.tight_layout()
        plt.show()

# MODEL MANAGER
class ModelManager:
    @staticmethod
    def save(model, path):
        torch.save(model.state_dict(), path)
        print(f"\nModel saved: {path}")

    @staticmethod
    def load(model, path):
        model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)
        model.eval()
        print(f"\nModel loaded: {path}")
        return model

# PREDICTOR
class IrisPredictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, sepal_length, sepal_width, petal_length, petal_width):
        data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        # Same scaler used during training
        data_scaled = (self.scaler.transform(data))
        tensor = torch.tensor(data_scaled, dtype=torch.float32).to(Config.DEVICE)
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = (torch.argmax(probabilities, dim=1).item())
            confidence = (probabilities[0, predicted_class].item())
        species = (Config.REVERSE_MAPPING[predicted_class])
        return species, confidence

# VISUALIZER
class TrainingVisualizer:
    @staticmethod
    def plot_loss(loss_history):
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_accuracy(accuracy_history):
        plt.figure(figsize=(8, 5))
        plt.plot(accuracy_history, label="Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()

# MAIN APPLICATION
class IrisClassificationApp:
    def __init__(self):
        self.data_loader = (IrisDataLoader(Config.CSV_PATH))
        self.preprocessor = (IrisPreprocessor())
        self.model = IrisClassifier()
        self.model.to(Config.DEVICE)

    def run(self):
        print("\n" + "=" * 60)
        print("PYTORCH IRIS CLASSIFICATION")
        print("OOP END-TO-END PROJECT")
        print("=" * 60)
        # 1. Load Dataset
        df = (self.data_loader.load())
        self.data_loader.show_info(df)
        # 2. Preprocessing
        (x_train, x_test, y_train, y_test) = (self.preprocessor.prepare_data(df))
        print("\nTrain Shape:", x_train.shape)
        print("Test Shape:", x_test.shape)
        # 3. PyTorch Dataset
        train_dataset = IrisDataset(x_train, y_train)
        test_dataset = IrisDataset(x_test, y_test)
        # 4. DataLoader
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        # 5. Show Model
        print("\n" + "=" * 60)
        print("MODEL ARCHITECTURE")
        print("=" * 60)
        print(self.model)
        # 6. Train
        trainer = IrisTrainer(self.model, train_loader)
        trainer.train()
        # 7. Evaluate
        evaluator = IrisEvaluator(self.model, test_loader)
        (y_true, y_pred, accuracy) = evaluator.evaluate()
        # 8. Confusion Matrix
        evaluator.plot_confusion_matrix(y_true, y_pred)
        # 9. Training Graphs
        TrainingVisualizer.plot_loss(trainer.loss_history)
        TrainingVisualizer.plot_accuracy(trainer.accuracy_history)
        # 10. Save Model
        ModelManager.save(self.model, Config.MODEL_PATH)
        # 11. Load Model
        loaded_model = (IrisClassifier())
        loaded_model = (ModelManager.load(loaded_model, Config.MODEL_PATH))
        # 12. New Prediction
        predictor = IrisPredictor(loaded_model, self.preprocessor.scaler)
        prediction, confidence = (predictor.predict(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2))
        print("\n" + "=" * 60)
        print("NEW DATA PREDICTION")
        print("=" * 60)
        print("\nInput:")
        print("Sepal Length : 5.1")
        print("Sepal Width  : 3.5")
        print("Petal Length : 1.4")
        print("Petal Width  : 0.2")
        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {confidence * 100:.2f}%")
        # Final
        print("\n" + "=" * 60)
        print("PROJECT COMPLETED")
        print("=" * 60)
        print(f"Final Accuracy: {accuracy * 100:.2f}%")

# ENTRY POINT
if __name__ == "__main__":
    SeedManager.set_seed(Config.RANDOM_STATE)
    app = IrisClassificationApp()
    app.run()
