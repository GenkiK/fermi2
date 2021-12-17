from __future__ import annotations
from fermi_model import *
from typing import Callable

from numpy import float64


def simpson(x_0: float, x_N: float, func: Callable, N: int = 10000) -> float:
    """シンプソン法による積分計算"""
    h = (x_N - x_0) / N
    S = 0
    for i in range(N // 2):
        S += h / 3 * (func(x_0 + 2 * i * h) + 4 * func(x_0 + (2 * i + 1) * h) + func(x_0 + (2 * i + 2) * h))
    return S


def approx_degeneracy_func(score: int, score_0: int) -> float:
    """エネルギー準位密度の近似式（ρ(ε) ∝ 1/(ε-E_0) * exp(√ε-E_0)）"""
    E = score - score_0
    if E == 0:  # 基底状態のエネルギー準位密度は 1 とする
        return 1
    return np.exp(2 * np.sqrt(np.pi ** 2 * E / 6)) / np.sqrt(48) / E


# 縮退度で平均するときに使用する
def calc_approx_degeneracies(scores: list[int]) -> list[float]:
    """近似した縮退度を計算"""
    score_0 = scores[0]
    return [approx_degeneracy_func(score, score_0) for score in scores]


def calc_approx_dist(scores: list[int], Te: float, ne: float) -> list[float]:
    approx_dist = [1]
    score_0 = scores[0]
    for score in scores[1:]:
        n = ne / Te ** 0.5 * np.exp(-(score - score_0) / Te)
        n /= simpson(score_0, score, lambda x: approx_degeneracy_func(x, score_0) * (score - x) ** 3)
        approx_dist.append(n)
    approx_dist[0] = 1 - sum(approx_dist[1:])
    return approx_dist


def calc_approx_mean_dist(scores: list[int], Te: float, ne: float) -> NDArray[float64]:
    approx_dist = calc_approx_dist(scores, Te, ne)
    return np.array(approx_dist) / np.array(calc_approx_degeneracies(scores))


def convert_eps2eV(val: float, t: float = 1.0) -> float:
    return val * 400 * np.pi ** 2 * t / 3


def convert_eV2eps(val: float, t: float = 1.0) -> float:
    return val * 3 / (400 * np.pi ** 2 * t)
