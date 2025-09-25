"""
Script: smoke_check
Propósito:
    Verificar que el entorno está correctamente configurado.
    - Imprime versiones de librerías clave.
    - Escribe y lee un CSV de prueba.
Salida esperada:
    Mensajes en consola indicando éxito de las operaciones.
"""

import sys
import os
import pandas as pd

def main() -> None:
    print("== Verificación de entorno ==")
    print(f"Python: {sys.version}")
    try:
        import numpy as np
        print(f"Numpy: {np.__version__}")
    except Exception:
        print("Numpy no disponible.")

    try:
        import networkx as nx
        print(f"NetworkX: {nx.__version__}")
    except Exception:
        print("NetworkX no disponible.")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except Exception:
        print("PyTorch no disponible.")

    # Prueba de E/S con CSV
    datos = {"columna_a": [1, 2, 3], "columna_b": ["x", "y", "z"]}
    df = pd.DataFrame(datos)
    ruta = os.path.join("data", "processed", "smoke_check.csv")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    df.to_csv(ruta, index=False)

    df_leido = pd.read_csv(ruta)
    print("CSV escrito y leído correctamente. Filas:", len(df_leido))

if __name__ == "__main__":
    main()
