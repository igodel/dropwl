"""
Módulo: timing
Propósito:
    Medir tiempos de ejecución de bloques de código de forma simple.
"""

from typing import Optional
import time


class Cronometro:
    """
    Clase:
        Cronometro

    Propósito:
        Medir el tiempo transcurrido entre 'iniciar' y 'detener'.

    Uso:
        c = Cronometro(nombre="ejemplo")
        c.iniciar()
        # ... bloque de código ...
        c.detener()
        print(c.tiempo_total_segundos)
    """

    def __init__(self, nombre: Optional[str] = None) -> None:
        self.nombre = nombre
        self._inicio = 0.0
        self._fin = 0.0
        self.tiempo_total_segundos = 0.0

    def iniciar(self) -> None:
        self._inicio = time.perf_counter()

    def detener(self) -> None:
        self._fin = time.perf_counter()
        self.tiempo_total_segundos = self._fin - self._inicio
