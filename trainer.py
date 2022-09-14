import time, copy
import os
from typing import Any, Dict, List, Union
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import *
import datetime

class Trainer():
    SAVE_FOLDER = "checkpoints"
    def __init__(self,
                 model: nn.Module,
                 dataloaders: Dict[str, DataLoader],
                 criterion: Union[torch.nn.MSELoss, torch.nn.CrossEntropyLoss, torch.nn.KLDivLoss, torch.nn.BCEWithLogitsLoss],
                 optimizer: Union[torch.optim.AdamW, torch.optim.Adam],
                 scheduler: Union[torch.optim.lr_scheduler.ReduceLROnPlateau, None],
                 verbose: bool = True) -> None:
        """ Le Trainer peut entraîner et tester un modèle sur les données contenues dans dataloaders.

        Args:
            model (nn.Module): Le modèle à entraîner.
            dataloaders (Dict[str, DataLoader]): Les données d'entraînement, de validation et (optionnellement) de test.
            criterion (Union[torch.nn.MSELoss, torch.nn.CrossEntropyLoss, torch.nn.KLDivLoss, torch.nn.BCEWithLogitsLoss]): La fonction de coût.
            optimizer (Union[torch.optim.AdamW, torch.optim.Adam]): L'optimiseur.
            scheduler (Union[torch.optim.lr_scheduler.ReduceLROnPlateau, None]): Le scheduler du learning_rate.
            verbose (bool, optional): Si on affiche plus que le tqdm. Defaults to True.

        Raises:
            TypeError: Si dataloaders n'est pas un dictionnaire.
            TypeError: Si dataloaders ne contient pas Train et Val dans ces clées
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not isinstance(dataloaders, dict) or "Train" not in dataloaders or "Val" not in dataloaders: 
            raise TypeError("dataloaders doit minimalement contenir {'Train': Dataloader, 'Val': Dataloader}")
        if not isinstance(dataloaders["Train"], DataLoader):
            raise TypeError(f"dataloaders['Train'] doit être un Dataloader. Pas un {type(dataloaders['Train'])}.")
        if not isinstance(dataloaders["Val"], DataLoader):
            raise TypeError(f"dataloaders['Val'] doit être un Dataloader. Pas un {type(dataloaders['Val'])}.")
        
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.verbose = verbose
        self.best_loss = np.inf
        self.metrics = {"Epoch": [], "Train_loss": [], "Val_loss": [], "Test_loss": []}
        
    def get_outputs(self,
                    inputs: torch.Tensor) -> torch.Tensor:
        """ Renvoie les logits du modèle sur les inputs de la batch (appel le forward du modèle).

        Args:
            inputs (torch.Tensor): Les inputs de la batch sur lesquels le modèle doit faire des prédictions.

        Raises:
            NotImplementedError: Cette méthode doit être implémentée par la classe fille.

        Returns:
            torch.Tensor: Les logits du modèle sur les inputs.
        """
        raise NotImplementedError("Cette méthode n'est pas implémentée.")
    
    def calculate_metrics(self,
                    outputs: torch.Tensor,
                    labels: torch.Tensor) -> None:
        """ Calcul les métriques et les ajoutent à self.metrics.

        Args:
            outputs (torch.Tensor): Les logits du modèle sur les inputs.
            labels (torch.Tensor): Les labels des inputs.

        Raises:
            NotImplementedError: Cette méthode doit être implémentée par la classe fille.

        """
        raise NotImplementedError("Cette méthode n'est pas implémentée.")
    
    def _run_epoch(self,
                   phase: str) -> None:
                   
        if phase == "Train":
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0
        for batch in tqdm(self.dataloaders[phase]):
            if not isinstance(batch, dict) or "input_ids" not in batch or "labels" not in batch:
                raise TypeError("un next(iter(dataloader)) doit retourner un dictionnaire de la forme {'input_ids': tensor, 'labels': tensor}. Pas {}.".format(batch))
            if not isinstance(batch["input_ids"], torch.Tensor):
                raise TypeError(f"La clée 'input_ids' ne retourne pas un tensor, elle retourne {type(batch['input_ids'])}.")
            if not isinstance(batch["labels"], torch.Tensor):
                raise TypeError(f"La clée 'labels' ne retourne pas un tensor, elle retourne {type(batch['labels'])}.")
            batch = {key: value.to(self.device) for key, value in batch.items()}
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "Train"):
                outputs = self.get_outputs(batch)
                loss = self.criterion(outputs, batch["labels"])
                if phase == "Train":
                    loss.backward()
                    self.optimizer.step()
            running_loss += loss.item() * outputs.size(0)
        self.metrics[f"{phase}_loss"].append(running_loss / len(self.dataloaders[phase]))
        if len(list(self.metrics.keys())) > 4:
            self.calculate_metrics(outputs, batch["labels"])
        if phase == "Val":
            if self.metrics["Val_loss"][-1] < self.best_loss:
                print("Val loss passe de {:.4f} à {:.4f}".format(self.best_loss, self.metrics["Val_loss"][-1]))
                self.best_loss = self.metrics["Val_loss"][-1]
                self.stagnation = 0
                if self.save_path is not None:
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    model_name = f"best_model_at_{self.when_training_started}.pt" if self.save_name is None else self.save_name
                    torch.save(best_model_wts, os.path.join(self.save_path, self.SAVE_FOLDER, model_name))
            elif self.patience:
                self.stagnation += 1
                if self.stagnation == self.patience:
                    print(f"Stagnation depuis {self.patience} epochs. Fin de l'entraînement.")
                    return
            if self.scheduler is not None:
                self.scheduler.step(self.metrics[f"{phase}_loss"][-1])
                
    def train(self,
              epochs: int = 10,
              save_folder_path: Union[str, None] = None,
              save_name: Union[str, None] = None,
              patience: Union[int, None] = None) -> Dict[str, List[float]]:
        
        """ Entraine le modèle sur les données d'entraînement et le valide sur les données de validation.

        Args:
            epochs (int): Le nombre d'epochs d'entraînement et de validation. Defaults to 10.
            metrics (Union[List[str], None]): Les métriques d'évaluation du modèle. Defaults to None.
            save_folder_path (Union[str, None]): Le path du dossier ou sera sauvegardé le meilleur modèle entrainé, si None il n'y a pas de sauvegarde. Defaults to None.
            save_name (Union[str, None]): Le nom du modèle entraîné, si None un nom par défaut lui sera attribué. Defaults to None.
            patience (Union[int, None]): Le nombre d'epochs sans amélioration de la loss de validation avant d'arrêter l'entraînement. Defaults to None.

        Raises:
            TypeError: Si dataloaders n'est pas un dictionnaire.
            TypeError: Si dataloaders ne contient pas Train et Val dans ces clées
            
        Returns:
            Dict[str, List[float]]: Les métriques calculées sur les données de Train et Val.
        """
        
        if not isinstance(epochs, int):
            raise TypeError("epochs doit être un entier.")
        if not isinstance(save_folder_path, (str, type(None))):
            raise TypeError("save_folder doit être une chaîne de caractères ou None.")
        
        if save_folder_path is not None:
            if not os.path.isdir(save_folder_path):
                raise ValueError("save_path doit être un chemin vers un dossier existant.")
            if not os.path.exists(save_folder_path):
                os.mkdir(save_folder_path)
            if not os.path.exists(os.path.join(save_folder_path, self.SAVE_FOLDER)):
                os.mkdir(os.path.join(save_folder_path, self.SAVE_FOLDER))
                
        self.when_training_started = datetime.datetime.now().strftime("%d-%m-%Y_%Hh%M")
        since = time.perf_counter()
        self.save_path = save_folder_path
        self.save_name = save_name
        self.stagnation = 0
        self.patience = patience
        for epoch in range(epochs):
            if self.verbose:
                print("-" * 30)
                print(f"Epoch {epoch+1}/{epochs}")
                print_memory_usage(self.device)
            self.metrics["Epoch"].append(epoch+1)
            for phase in ["Train", "Val"]:
                self._run_epoch(phase)
               
        duree = time.perf_counter() - since 
        if self.verbose:
            print(f"Durée du Training {duree//60:.0f} min {duree%60:.0f} s")
        return self.metrics
    
    def test(self, test_dataloader: DataLoader) -> Dict[str, List[float]]:
        """Teste le modèle sur les données de test du dataloader passé en argument.

        Args:
            metrics (Union[List[str], None]): Liste des métriques à calculer.

        Raises:
            TypeError: Si Test n'est pas dans les clées du dataloaders.

        Returns:
            Dict[str, List[float]]: Les métriques calculées sur les données de Test.
        """

        if not isinstance(test_dataloader, DataLoader):
            raise TypeError(f"test_dataloader doit être un Dataloader. Pas un {type(test_dataloader)}.")

        self.dataloaders["Test"] = test_dataloader
        self._run_epoch("Test")
        return self.metrics
    
    def add_metrics_keys(self, metrics: Union[List[str], None]) -> None:
        """ Ajoute les métriques de la liste comme nouvelles clées au dict self.metrics du trainer.

        Args:
            new_metrics (Union[List[str], None]): Liste des nouvelles métriques à ajouter à self.metrics.

        Raises:
            TypeError: Si metrics n'est pas une liste.
            KeyError: Si une des métiques de metrics est déjà présente dans self.metrics.
        """
        
        if isinstance(metrics, list):
            for metric in metrics:
                if metric in self.metrics:
                    raise KeyError(f"La métrique {metric} est déjà contenue dans les métriques du trainer. Soit, {self.metrics.keys()}.")
                else:
                    self.metrics[str(metric)] = []
        elif metrics is not None:
            raise TypeError(f"Les nouvelles métrics doivent êtres une liste ou None, pas {type(metrics)}.")
