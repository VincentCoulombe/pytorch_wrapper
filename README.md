# pytorch_wrapper
![Tests](https://github.com/VincentCoulombe/pytorch_wrapper/actions/workflows/tests.yml/badge.svg)
## Comment l'installer?
1. S'assurer d'avoir les requirements d'installés.
2. `pip install git+https://github.com/VincentCoulombe/pytorch_wrapper`
3. `from pytorch_wrapper.trainer import Trainer`

## Trainer  

### 1. Créer un enfant de la classe parent Trainer et override les méthodes get_preds_and_loss() et calculate_metrics()
```
class TestTrainer(Trainer):
    def __init__(self,
                 model: nn.Module,
                 dataloaders,
                 optimizer,
                 scheduler = None,
                 verbose: bool = True) -> None:
        """ Initialise la classe Trainer.

        Args:
            model (nn.Module): Un Modèle PyTorch.
            dataloaders (Dict[str: Dataloaders]): Un dictionnaire contenant MINIMALEMENT {'Train': Dataloader, 'Val': Dataloader}
            optimizer (Any): Un optimiseur PyTorch.
            scheduler (Any, optional): Un scheduler PyTorch. Defaults to None.
            verbose (bool, optional): Verbose. Defaults to True.
        """
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
                    predictions: list,
                    labels: list,
                    phase: str) -> None:
        """ Calcul les métriques et les ajoutent à self.metrics.

        Args:
            predictions (torch.Tensor): Les prédictions du modèle sur tous les inputs d'un epoch.
            labels (torch.Tensor): Les labels de tous les inputs d'un epoch.
            phase (str): La phase d'entraînement (Train, Val ou Test).

        Raises:
            NotImplementedError: Cette méthode doit être implémentée par la classe fille.

        """
        if phase != "Test":
            return
        if "Test_accuracy" in list(self.metrics.keys()):
            self.metrics["Test_accuracy"].append(accuracy_score(predictions, labels))
        if "Test_F1" in list(self.metrics.keys()):
            self.metrics["Test_F1"].append(f1_score(predictions, labels))
        if "Test_recall" in list(self.metrics.keys()):
            self.metrics["Test_recall"].append(recall_score(predictions, labels))
        if "Test_predictions" in list(self.metrics.keys()):
            self.metrics["Test_predictions"].extend(predictions)
        if "Test_labels" in list(self.metrics.keys()):
            self.metrics["Test_labels"].extend(labels)
```

### 2. Instancier un trainer
```
dataloaders = {'Train': train_dataloader, 'Val': val_dataloader}
optimizer = torch.optim.AdamW(model.parameters())
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

trainer = ClassificationTrainer(model, dataloaders, optimizer, scheduler)
```

### 3. Ajouter des métriques qui seront calculées dans calculate_metrics

Les métriques de base sont : Epoch, Train_loss, Val_loss, Test_loss.
```
trainer.add_metrics_keys(["Test_accuracy", "Test_F1", "Test_recall"])
```

### 4. Lancer l'entraînement!

Retourne un dict contenant les métriques (Epoch, Train_loss, Val_loss) pour chaque époques.
```
metrics = trainer.train(epochs=2)
```

Posibilité de sauvegarder le meilleur modèle en validation en spécifiant un path.
```
metrics = trainer.train(epochs=10, save_folder_path=".")
```

Posibilité de spécifier le nom du modèle sauvegardé (défault = la date et l'heure du début du training).
```
metrics = trainer.train(epochs=10, save_folder_path=".", save_name="test_model.pt")
```

Posibilité de faire du early-stopping en passant une patience en paramètre.
```
metrics = trainer.train(epochs=10, save_folder_path=".", save_name="test_model.pt", patience=2)
```
 
### 5. Tester l'entraînement!

Retourne un dict contenant les métriques (Epoch, Train_loss, Val_loss) pour chaque époques ET un Test_loss.
```
metrics = trainer.test(DataLoader(TestDataset(), batch_size=1, shuffle=True, pin_memory=True)))
```


