# pytorch_wrapper

## Comment l'installer?
1. S'assurer d'avoir les requirements d'installés.
2. `pip install git+https://github.com/VincentCoulombe/pytorch_wrapper`
3. `from pytorch_wrapper import *`

## Trainer  

### 1. Instancier un enfant de la classe et override les méthodes get_outputs et calculate_metrics
```
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
                    labels: torch.Tensor) -> None:
        """ Calcul les métriques et les ajoutent à self.metrics.

        Args:
            predictions (torch.Tensor): Les prédictions du modèle sur les inputs.
            labels (torch.Tensor): Les labels des inputs.

        Raises:
            NotImplementedError: Cette méthode doit être implémentée par la classe fille.

        """
        usable_predictions = predictions.detach().cpu().numpy()
        usable_labels = labels.detach().cpu().numpy()
        if "Test_accuracy" in list(self.metrics.keys()):
            self.metrics["Test_accuracy"].append(accuracy_score(usable_predictions, usable_labels))
        if "Test_F1" in list(self.metrics.keys()):
            self.metrics["Test_F1"].append(f1_score(usable_predictions, usable_labels))
        if "Test_recall" in list(self.metrics.keys()):
            self.metrics["Test_recall"].append(recall_score(usable_predictions, usable_labels))
```

### 2. Ajouter des métriques qui seront calculés dans calculate_metrics

Les métriques de base sont : Epoch, Train_loss, Val_loss, Test_loss
```
trainer.add_metrics_keys(["Test_accuracy", "Test_F1", "Test_recall"])
```

### 3. Lancer l'entraînement!

Retourne un dict contenant les métriques pour chaque époques.
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

Posibilité de faire du early-stopping en passant une patiance en paramètre.
```
metrics = trainer.train(epochs=10, patience=2)
```
 
### 4. Tester l'entraînement!

Retourne un dict contenant les métriques pour chaque époques ET celles de test.
```
metrics = trainer.test(DataLoader(TestDataset(), batch_size=1, shuffle=True, pin_memory=False)))
```


