import numpy as np


def error_absoluto(aproximado: float, exacto: float) -> float:
    return abs(aproximado - exacto)


def error_relativo(aproximado: float, exacto: float) -> float:
    if exacto == 0:
        return float("inf") if aproximado != 0 else 0.0
    return abs((aproximado - exacto) / exacto)


def intervalo_confianza_95(valores: np.ndarray) -> tuple[float, float, float]:
    media = np.mean(valores)
    std = np.std(valores, ddof=1) if len(valores) > 1 else 0.0
    margen = 1.96 * std / np.sqrt(len(valores))
    return float(media), float(media - margen), float(media + margen)
