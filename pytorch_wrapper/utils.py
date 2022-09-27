import torch
from typing import Dict, List, Union

def print_memory_usage(device) -> None:
    
    if device == torch.device("cuda"):
        memoire_utilisee, memoire_disponible = torch.cuda.mem_get_info()
        if memoire_disponible == 0:
            print("Aucune mémoire GPU de disponible.")
        else:
            print(f"{memoire_utilisee * 100 / memoire_disponible:.02f}% de la mémoire GPU est utilisée.")
        
