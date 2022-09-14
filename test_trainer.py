import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score

from trainer import *

class TestModel(nn.Module):
    
    def __init__(self, in_features=1, h1=1, h2=1, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = self.out(x)
        return x


class TestDataset(Dataset):
    def __init__(self):
        self.labels = torch.randint(0, 1, (24,1), dtype=torch.float32)
        self.input_ids = torch.randint(0, 1, (24,1), dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        labels = self.labels[index]
        input_ids = self.input_ids[index]
        return {"input_ids": input_ids, "labels": labels}
    
class TestTrainer(Trainer):
    def __init__(self,
                model,
                dataloaders,
                criterion,
                optimizer,
                scheduler,
                verbose) -> None:
        super().__init__(model, dataloaders, criterion, optimizer, scheduler, verbose)
        
    def get_outputs(self,
                    inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs["input_ids"])
    
    
    def calculate_metrics(self,
                outputs: torch.Tensor,
                labels: torch.Tensor) -> None:
        usable_outputs = outputs.detach().cpu().numpy().astype(int)
        usable_labels = labels.detach().cpu().numpy().astype(int)
        if "Test_accuracy" in list(self.metrics.keys()):
            self.metrics["Test_accuracy"].append(accuracy_score(usable_outputs, usable_labels))
        if "Test_F1" in list(self.metrics.keys()):
            self.metrics["Test_F1"].append(f1_score(usable_outputs, usable_labels))
        if "Test_recall" in list(self.metrics.keys()):
            self.metrics["Test_recall"].append(recall_score(usable_outputs, usable_labels))
    
def test_creer_valid():
    
    for scheduler in [VALID_SCHEDULER, None]:
        trainer = TestTrainer(VALID_MODEL,
                        VALID_LOADERS,
                        VALID_CRITERION,
                        VALID_OPTIMIZER,
                        scheduler,
                        True)
        assert isinstance(trainer, Trainer)
        assert isinstance(trainer.model, nn.Module)
        assert isinstance(trainer.dataloaders, dict)
        assert isinstance(trainer.criterion, (torch.nn.MSELoss, torch.nn.CrossEntropyLoss, torch.nn.KLDivLoss, torch.nn.BCEWithLogitsLoss))
        assert isinstance(trainer.optimizer, (torch.optim.AdamW, torch.optim.Adam))
        assert isinstance(trainer.scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, type(None)))
        assert isinstance(trainer.verbose, bool)
    
def test_creer_invalid():
    for invalid_loader in [INVALID_KEY_LOADERS, INVALID_TRAIN_LOADER, INVALID_VAL_LOADER]:
        with pytest.raises(TypeError):
            trainer = TestTrainer(VALID_MODEL,
                              invalid_loader,
                              VALID_CRITERION,
                              VALID_OPTIMIZER,
                              VALID_SCHEDULER,
                              True)
            
def test_train_valid():
    trainer = TestTrainer(VALID_MODEL,
                USABLE_LOADERS,
                VALID_CRITERION,
                VALID_OPTIMIZER,
                None,
                True)
    
    trainer.train(epochs=2)
    assert len(trainer.metrics["Val_loss"]) == 2
    assert len(trainer.metrics["Train_loss"]) == 2
    assert len(trainer.metrics["Epoch"]) == 2
    assert len(trainer.metrics["Test_loss"]) == 0
    
    trainer = TestTrainer(VALID_MODEL,
            USABLE_LOADERS,
            VALID_CRITERION,
            VALID_OPTIMIZER,
            VALID_SCHEDULER,
            True)
    
    trainer.train(epochs=10, save_folder_path=".", save_name="test_model.pt")
    assert len(trainer.metrics["Val_loss"]) == 10
    assert len(trainer.metrics["Train_loss"]) == 10
    assert len(trainer.metrics["Epoch"]) == 10
    assert len(trainer.metrics["Test_loss"]) == 0
    model = VALID_MODEL
    model.load_state_dict(torch.load(os.path.join(".", "checkpoints", "test_model.pt")))
    
    new_trainer = TestTrainer(model,
        USABLE_LOADERS,
        VALID_CRITERION,
        VALID_OPTIMIZER,
        VALID_SCHEDULER,
        True)
    
    new_trainer.train(epochs=25, save_folder_path=None)
    assert len(new_trainer.metrics["Val_loss"]) == 25
    assert len(new_trainer.metrics["Train_loss"]) == 25
    assert len(new_trainer.metrics["Epoch"]) == 25
    assert len(new_trainer.metrics["Test_loss"]) == 0
    assert new_trainer.best_loss < trainer.best_loss
    
def test_add_metrics_keys_valid():
    trainer = TestTrainer(VALID_MODEL,
            USABLE_LOADERS,
            VALID_CRITERION,
            VALID_OPTIMIZER,
            None,
            True)
    
    assert list(trainer.metrics.keys()) == ["Epoch", "Train_loss", "Val_loss", "Test_loss"]
    trainer.add_metrics_keys(["Test_accuracy"])
    assert list(trainer.metrics.keys()) == ["Epoch", "Train_loss", "Val_loss", "Test_loss", "Test_accuracy"]
    trainer.add_metrics_keys(["Test_F1", "Test_recall"])
    assert list(trainer.metrics.keys()) == ["Epoch", "Train_loss", "Val_loss", "Test_loss", "Test_accuracy", "Test_F1", "Test_recall"]
    trainer.add_metrics_keys(None)
    assert list(trainer.metrics.keys()) == ["Epoch", "Train_loss", "Val_loss", "Test_loss", "Test_accuracy", "Test_F1", "Test_recall"]
    trainer.add_metrics_keys([1])
    assert list(trainer.metrics.keys()) == ["Epoch", "Train_loss", "Val_loss", "Test_loss", "Test_accuracy", "Test_F1", "Test_recall", "1"]
    
def test_add_metrics_keys_invalid():
    trainer = TestTrainer(VALID_MODEL,
            USABLE_LOADERS,
            VALID_CRITERION,
            VALID_OPTIMIZER,
            None,
            True)
    
    with pytest.raises(TypeError):
        trainer.add_metrics_keys(1)
        trainer.add_metrics_keys("Test_accuracy")
        
    trainer.add_metrics_keys(["Test_accuracy"])
    with pytest.raises(KeyError):
        trainer.add_metrics_keys(["Test_accuracy"])
    
def test_test_valid():
    model = VALID_MODEL
    model.load_state_dict(torch.load(os.path.join(".", "checkpoints", "test_model.pt")))
    
    trainer = TestTrainer(model,
        USABLE_LOADERS_W_TEST,
        VALID_CRITERION,
        VALID_OPTIMIZER,
        VALID_SCHEDULER,
        True)
    
    assert list(trainer.metrics.keys()) == ["Epoch", 'Train_loss', 'Val_loss', 'Test_loss']
    trainer.add_metrics_keys(["Test_accuracy"])
    trainer.test(DataLoader(TestDataset(), 8))
    assert list(trainer.metrics.keys()) == ["Epoch", 'Train_loss', 'Val_loss', 'Test_loss', 'Test_accuracy']
    assert len(trainer.metrics["Test_accuracy"]) == 1
    assert len(trainer.metrics["Test_loss"]) == 1
    assert len(trainer.metrics["Val_loss"]) == 0
    assert len(trainer.metrics["Train_loss"]) == 0
    assert len(trainer.metrics["Epoch"]) == 0
    trainer.add_metrics_keys(["Test_F1", "Test_recall"])
    trainer.test(DataLoader(TestDataset(), 8))
    assert list(trainer.metrics.keys()) == ["Epoch", 'Train_loss', 'Val_loss', 'Test_loss', 'Test_accuracy', "Test_F1", "Test_recall"]
    assert len(trainer.metrics["Test_F1"]) == 1
    assert len(trainer.metrics["Test_recall"]) == 1
    assert len(trainer.metrics["Test_accuracy"]) == 2
    assert len(trainer.metrics["Test_loss"]) == 2
    assert len(trainer.metrics["Val_loss"]) == 0
    assert len(trainer.metrics["Train_loss"]) == 0
    assert len(trainer.metrics["Epoch"]) == 0
    

VALID_MODEL = TestModel()
VALID_LOADERS = {"Train": DataLoader(torch.rand(10, 4), 8), "Val": DataLoader(torch.rand(10, 4), 8)}
USABLE_LOADERS = {"Train": DataLoader(TestDataset(), 8), "Val": DataLoader(TestDataset(), 8)} 
USABLE_LOADERS_W_TEST = {"Train": DataLoader(TestDataset(), 8), "Val": DataLoader(TestDataset(), 8), "Test": DataLoader(TestDataset(), 8)} 
INVALID_KEY_LOADERS = {"Train": DataLoader(torch.rand(10, 4), 8), "Test": DataLoader(torch.rand(10, 4), 8)}
INVALID_TRAIN_LOADER = {"Train": torch.rand(10, 4), "Val": DataLoader(torch.rand(10, 4), 8)}
INVALID_VAL_LOADER = {"Train": DataLoader(torch.rand(10, 4), 8), "Val": torch.rand(10, 4)}
VALID_CRITERION = torch.nn.MSELoss()
VALID_OPTIMIZER = torch.optim.AdamW(VALID_MODEL.parameters())
VALID_SCHEDULER  = torch.optim.lr_scheduler.ReduceLROnPlateau(VALID_OPTIMIZER)


test_test_valid()