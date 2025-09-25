"""
Módulo: io
Propósito:
    Lectura y escritura de datos en formatos simples (CSV, pickle).
Estilo:
    Funciones explícitas, sin azúcar sintáctica, autocontenidas.
"""

from typing import Any, Dict
import os
import pickle
import pandas as pd


def asegurar_directorio(ruta_directorio: str) -> None:
    """
    Propósito:
        Crear un directorio si no existe.

    Parámetros:
        ruta_directorio:
            Ruta del directorio a crear.

    Retorno:
        None
    """
    if os.path.isdir(ruta_directorio) is False:
        os.makedirs(ruta_directorio, exist_ok=True)


def guardar_csv(df: pd.DataFrame, ruta_csv: str, index: bool = False) -> None:
    """
    Propósito:
        Guardar un DataFrame en formato CSV.

    Parámetros:
        df:
            DataFrame de pandas a guardar.
        ruta_csv:
            Ruta completa del archivo CSV.
        index:
            Si es True, incluye el índice del DataFrame en el archivo.

    Retorno:
        None
    """
    directorio = os.path.dirname(ruta_csv)
    if directorio != "":
        asegurar_directorio(directorio)
    df.to_csv(ruta_csv, index=index)


def leer_csv(ruta_csv: str) -> pd.DataFrame:
    """
    Propósito:
        Leer un archivo CSV en un DataFrame de pandas.

    Parámetros:
        ruta_csv:
            Ruta completa del archivo CSV.

    Retorno:
        DataFrame con el contenido leído.
    """
    df = pd.read_csv(ruta_csv)
    return df


def guardar_pickle(objeto: Any, ruta_pickle: str) -> None:
    """
    Propósito:
        Guardar un objeto Python en formato pickle (binario).

    Parámetros:
        objeto:
            Objeto serializable con pickle.
        ruta_pickle:
            Ruta completa del archivo destino.

    Retorno:
        None
    """
    directorio = os.path.dirname(ruta_pickle)
    if directorio != "":
        asegurar_directorio(directorio)
    with open(ruta_pickle, "wb") as f:
        pickle.dump(objeto, f, protocol=pickle.HIGHEST_PROTOCOL)


def leer_pickle(ruta_pickle: str) -> Any:
    """
    Propósito:
        Leer un objeto Python desde un archivo pickle.

    Parámetros:
        ruta_pickle:
            Ruta completa del archivo origen.

    Retorno:
        Objeto deserializado.
    """
    with open(ruta_pickle, "rb") as f:
        objeto = pickle.load(f)
    return objeto
