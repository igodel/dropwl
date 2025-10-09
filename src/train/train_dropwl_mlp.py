"""
Módulo: train_dropwl_mlp
Propósito:
    Entrenar y evaluar el modelo dropWL + MLP para grafos pequeños.
    Por época y por grafo se muestrean R ejecuciones (dropout -> 1-WL -> firma).
    Cada firma pasa por el MLP compartido y se agrega por promedio.
    Un cabezal lineal produce logits para clasificación.

Estilo:
    - Código académico, explícito, sin azúcar sintáctica.
    - CPU suficiente para grafos pequeños.
"""

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx

from src.core.dropout import sample_node_mask, apply_node_dropout
from src.core.wl_refinement import compute_wl_colors
from src.signatures.graph_signatures import construir_firma_histograma
from src.mlp.mlp_head import MLPHead


def representar_grafo_dropwl_mlp_once(
    grafo: nx.Graph,
    p: float,
    R: int,
    t: int,
    k_max: int,
    semilla_base: int,
    mlp: nn.Module,
    device: torch.device
) -> torch.Tensor:
    """
    Propósito:
        Obtener representación Φ(G) aplicando el MLP por ejecución y luego promediando.

    Procedimiento:
        - Para i=0..R-1:
            * muestrear máscara con semilla_base + i
            * subgrafo inducido
            * 1-WL t iteraciones -> colores
            * firma φ_i (k_max)
            * h_i = MLP(φ_i)
        - Φ(G) = (1/R) * sum_i h_i

    Retorno:
        Tensor 1D de tamaño d (dimensión de salida del MLP).
    """
    acumulador = None
    i = 0
    while i < R:
        semilla = semilla_base + i

        # Dropout y subgrafo
        mascara = sample_node_mask(grafo, p, semilla)
        subgrafo = apply_node_dropout(grafo, mascara)

        # WL y firma
        colores = compute_wl_colors(subgrafo, numero_de_iteraciones=t, atributo_color_inicial=None)
        firma_np = construir_firma_histograma(colores, k_max)  # np.ndarray (k_max,)
        firma = torch.from_numpy(firma_np).float().to(device)  # tensor (k_max,)

        # MLP por ejecución
        h = mlp(firma)  # (d,)

        if acumulador is None:
            acumulador = torch.zeros_like(h)
        acumulador = acumulador + h
        i = i + 1

    representacion = acumulador / float(R)
    return representacion  # (d,)


def entrenar_epoch(
    grafos: List[nx.Graph],
    labels: np.ndarray,
    p: float,
    R: int,
    t: int,
    k_max: int,
    semilla_base: int,
    mlp: nn.Module,
    head: nn.Module,
    opt: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Propósito:
        Entrenar una época completa (forward-backward-update) con re-muestreo de ejecuciones.

    Retorno:
        Pérdida promedio (float) de la época.
    """
    mlp.train()
    head.train()

    criterio = nn.CrossEntropyLoss()
    total_loss = 0.0
    n = len(grafos)

    i = 0
    while i < n:
        g = grafos[i]
        y = int(labels[i])

        # Re-muestreo para esta época y este grafo
        phi = representar_grafo_dropwl_mlp_once(
            grafo=g, p=p, R=R, t=t, k_max=k_max,
            semilla_base=semilla_base + 1000 * i,
            mlp=mlp, device=device
        )  # (d,)

        logits = head(phi.unsqueeze(0))  # (1, num_clases)
        target = torch.tensor([y], dtype=torch.long, device=device)

        loss = criterio(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss = total_loss + float(loss.item())
        i = i + 1

    loss_prom = total_loss / float(n) if n > 0 else 0.0
    return loss_prom


def evaluar(
    grafos: List[nx.Graph],
    labels: np.ndarray,
    p: float,
    R: int,
    t: int,
    k_max: int,
    semilla_base: int,
    mlp: nn.Module,
    head: nn.Module,
    device: torch.device
) -> float:
    """
    Propósito:
        Evaluar accuracy con el mismo pipeline (mismos p, R, t, k_max).
        Se re-muestrea con semilla fija para reproducibilidad.
    """
    mlp.eval()
    head.eval()

    aciertos = 0
    n = len(grafos)

    with torch.no_grad():
        i = 0
        while i < n:
            g = grafos[i]
            y = int(labels[i])

            phi = representar_grafo_dropwl_mlp_once(
                grafo=g, p=p, R=R, t=t, k_max=k_max,
                semilla_base=semilla_base + 1000 * i,
                mlp=mlp, device=device
            )
            logits = head(phi.unsqueeze(0))  # (1, num_clases)
            pred = int(torch.argmax(logits, dim=1).item())

            if pred == y:
                aciertos = aciertos + 1
            i = i + 1

    accuracy = float(aciertos) / float(n) if n > 0 else 0.0
    return accuracy
