"""
Módulo: seed
Propósito:
    Fijar la semilla global para obtener resultados reproducibles.
Estilo:
    Código explícito, tipado y con comentarios breves de intención.
"""

from typing import Optional
import random
import os

def set_global_seed(valor_semilla: int, hacer_determinista_torch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(valor_semilla)
    random.seed(valor_semilla)
    try:
        import numpy as np  # type: ignore
        np.random.seed(valor_semilla)
    except ImportError:
        pass

    if hacer_determinista_torch:
        try:
            import torch  # type: ignore
            torch.manual_seed(valor_semilla)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(valor_semilla)
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False  # type: ignore
        except ImportError:
            pass
