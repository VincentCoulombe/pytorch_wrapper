import torch
from typing import Dict, List, Union

def print_memory_usage(device) -> None:
    
    if device == torch.device("cuda"):
        memoire_disponible = round(torch.cuda.max_memory_reserved()*1e-9,3)
        memoire_utilisee = round(torch.cuda.max_memory_allocated()*1e-9,3)
        print(f"{memoire_utilisee * 100 / memoire_disponible:.02f}% de la mémoire GPU est utilisée.")
        
