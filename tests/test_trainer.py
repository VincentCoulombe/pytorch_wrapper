import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score
import os

from pytorch_wrapper.trainer import *

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
                 model: nn.Module,
                 dataloaders: Dict[str, DataLoader],
                 optimizer = torch.optim.AdamW,
                 scheduler = None,
                 verbose: bool = True) -> None:
        super().__init__(model, dataloaders, optimizer, scheduler, verbose)
        
    def get_preds_and_loss(self,
                           batch: dict) -> tuple:
        """ Retourne les prédictions du modèle et la loss en fonction des inputs de la batch (appel le forward du modèle).

        Args:
            batch (dict): Une batch du DataLoader, minimalement de la forme {'input_ids': tensor, 'labels': tensor}.

        Raises:
            NotImplementedError: Cette méthode doit être implémentée par la classe fille.

        Returns:
            tuple: (prédictions: torch.Tensor, loss: torch.Tensor).
        """
        outputs = self.model(batch['input_ids'])
        predictions = torch.round(torch.sigmoid(outputs)) 
        criterion = nn.CrossEntropyLoss()
        return predictions, criterion(outputs, batch["labels"])
    
    
    def calculate_metrics(self,
                    predictions: torch.Tensor,
                    labels: torch.Tensor,
                    phase: str) -> None:
        """ Calcul les métriques et les ajoutent à self.metrics.

        Args:
            predictions (torch.Tensor): Les prédictions du modèle sur les inputs.
            labels (torch.Tensor): Les labels des inputs.
            phase (str): La phase d'entraînement (Train, Val ou Test).

        Raises:
            NotImplementedError: Cette méthode doit être implémentée par la classe fille.

        """
        if phase == "Test":
            usable_predictions = predictions.detach().cpu().numpy()
            usable_labels = labels.detach().cpu().numpy()
            if "Test_accuracy" in list(self.metrics.keys()):
                self.metrics["Test_accuracy"].append(accuracy_score(usable_predictions, usable_labels))
            if "Test_F1" in list(self.metrics.keys()):
                self.metrics["Test_F1"].append(f1_score(usable_predictions, usable_labels))
            if "Test_recall" in list(self.metrics.keys()):
                self.metrics["Test_recall"].append(recall_score(usable_predictions, usable_labels))
    
def test_creer_valid():
    
    for scheduler in [VALID_SCHEDULER, None]:
        trainer = TestTrainer(VALID_MODEL,
                        VALID_LOADERS,
                        VALID_OPTIMIZER,
                        scheduler,
                        True)
        assert isinstance(trainer, Trainer)
        assert isinstance(trainer.model, nn.Module)
        assert isinstance(trainer.dataloaders, dict)
        assert isinstance(trainer.optimizer, (torch.optim.AdamW, torch.optim.Adam))
        assert isinstance(trainer.scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, type(None)))
        assert isinstance(trainer.verbose, bool)
    
def test_creer_invalid():
    for invalid_loader in [INVALID_KEY_LOADERS, INVALID_TRAIN_LOADER, INVALID_VAL_LOADER]:
        with pytest.raises(TypeError):
            trainer = TestTrainer(VALID_MODEL,
                              invalid_loader,
                              VALID_OPTIMIZER,
                              VALID_SCHEDULER,
                              True)
            
def test_train_valid():
    
    if os.path.exists(os.path.join(".", "checkpoints", "test_model.pt")):
        os.remove(os.path.join(".", "checkpoints", "test_model.pt"))
        
    trainer = TestTrainer(VALID_MODEL,
                USABLE_LOADERS,
                VALID_OPTIMIZER,
                None,
                True)
    
    trainer.train(epochs=2)
    assert len(trainer.metrics["Val_loss"]) == 2
    assert len(trainer.metrics["Train_loss"]) == 2
    assert len(trainer.metrics["Epoch"]) == 2
    assert len(trainer.metrics["Test_loss"]) == 0
    assert not os.path.exists(os.path.join(".", "checkpoints", "test_model.pt"))
    
    trainer = TestTrainer(VALID_MODEL,
            USABLE_LOADERS,
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
        VALID_OPTIMIZER,
        VALID_SCHEDULER,
        True)
    
    metrics = new_trainer.train(epochs=25, patience=5)
    assert len(new_trainer.metrics["Val_loss"]) == 6
    assert len(new_trainer.metrics["Train_loss"]) == 6
    assert len(new_trainer.metrics["Epoch"]) == 6
    assert len(new_trainer.metrics["Test_loss"]) == 0
    
    metrics = new_trainer.train(epochs=25, patience=2)
    assert len(new_trainer.metrics["Val_loss"]) == 8
    assert len(new_trainer.metrics["Train_loss"]) == 8
    assert len(new_trainer.metrics["Epoch"]) == 8
    assert len(new_trainer.metrics["Test_loss"]) == 0
    
     
def test_add_metrics_keys_valid():
    trainer = TestTrainer(VALID_MODEL,
            USABLE_LOADERS,
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
VALID_OPTIMIZER = torch.optim.AdamW(VALID_MODEL.parameters())
VALID_SCHEDULER  = torch.optim.lr_scheduler.ReduceLROnPlateau(VALID_OPTIMIZER)
