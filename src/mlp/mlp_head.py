"""
Módulo: mlp_head
Propósito:
    Definir un MLP simple que mapea la firma de dimensión k_max a un espacio latente de dimensión d.
    Este MLP se aplica por ejecución (compartiendo pesos), antes de la agregación por media.

Estilo:
    - Código explícito y comentado.
    - Sin azúcar sintáctica.
    - Funciona con Python 3.8 y PyTorch CPU.
"""

from typing import Optional
import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """
    Clase:
        MLPHead

    Propósito:
        Implementar un perceptrón multicapa sencillo con 1 o 2 capas ocultas.
        Entrada:  k_max
        Salida:   d  (dimensión latente por ejecución)

    Parámetros:
        input_dim:  dimensión de entrada (k_max)
        hidden_dim: tamaño de la capa oculta (ej.: 128)
        output_dim: dimensión de salida (d)
        num_layers: número de capas (1 = lineal directa a d; 2 = una oculta + salida)
        activation: 'relu' o 'tanh'
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: str = "relu"
    ) -> None:
        super().__init__()

        if activation == "relu":
            act = nn.ReLU()
        else:
            act = nn.Tanh()

        if num_layers <= 1:
            # Mapeo lineal simple
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim)
            )
        else:
            # Una capa oculta + activación + salida
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                act,
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parámetros:
            x: tensor de forma (..., input_dim)
        Retorno:
            tensor de forma (..., output_dim)
        """
        y = self.net(x)
        return y
