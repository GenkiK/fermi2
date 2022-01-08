from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from itertools import combinations

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


def calc_approx_dist_eV(scores_eV: list[float], Te_eV: float, ne_eV: float) -> list[float]:
    approx_dist = [1]
    score_0 = scores_eV[0]
    C_ = 1.068e-14
    A_ = 106553.894
    for score in scores_eV[1:]:
        excitation = ne_eV * C_ / Te_eV ** 0.5 * np.exp(-(score - score_0) / Te_eV)
        n = excitation / simpson(score_0, score, lambda x: approx_degeneracy_func(x, score_0) * A_ * (score - x) ** 3)
        approx_dist.append(n)
    approx_dist[0] = 1 - sum(approx_dist[1:])
    return approx_dist


def calc_approx_mean_dist_eV(scores_eV: list[float], Te_eV: float, ne_eV: float) -> list[float]:
    approx_dist = calc_approx_dist_eV(scores_eV, Te_eV, ne_eV)
    return np.array(approx_dist) / np.array(calc_approx_degeneracies(scores_eV))


def outflux(score_eV: float, score_0: float, score_max: float, Te_eV: float, ne_eV: float) -> int:
    C_ = 1.068e-14
    A_ = 106553.894
    C_out = ne_eV * C_ / Te_eV ** 0.5 * simpson(score_eV, score_max, lambda x: approx_degeneracy_func(x, score_0) * np.exp(-(x - score_eV) / Te_eV))
    F_out = ne_eV * C_ / Te_eV ** 0.5 * simpson(score_0, score_eV, lambda x: approx_degeneracy_func(x, score_0))
    A_out = A_ * simpson(score_0, score_eV, lambda x: approx_degeneracy_func(x, score_0) * (score_eV - x) ** 3)
    return C_out + F_out + A_out


def influx(score_eV, partial_scores_eV: list[float], partial_dist: list[float], Te_eV: float, ne_eV: float):
    score_0 = partial_scores_eV[0]
    total_influx = 0
    for x, n in zip(partial_scores_eV, partial_dist):
        total_influx += approx_degeneracy_func(x, score_0) * n * ne_eV / Te_eV ** 0.5 * np.exp(-(score_eV - x) / Te_eV)
    return total_influx


def calc_approx_dist_eV_recursive(scores_eV: list[float], Te_eV: float, ne_eV: float):
    approx_dist = [1]
    score_0 = scores_eV[0]
    score_max = scores_eV[-1]

    for i, score in enumerate(scores_eV[1:]):
        n = influx(score, scores_eV[: i + 1], approx_dist, Te_eV, ne_eV) / outflux(score, score_0, score_max, Te_eV, ne_eV)
        approx_dist.append(n)
    approx_dist[0] = 1 - sum(approx_dist[1:])
    return approx_dist


def calc_approx_mean_dist_eV_recursive(scores_eV: list[float], Te_eV: float, ne_eV: float) -> list[float]:
    approx_dist = calc_approx_dist_eV_recursive(scores_eV, Te_eV, ne_eV)
    return np.array(approx_dist) / np.array(calc_approx_degeneracies(scores_eV))


# 解析的に発光強度を求める
def calc_approx_intensities(scores: list[float], Te: float, alpha: float) -> tuple[list[float], list[float]]:
    I_lst: list[float | None] = []
    dE_lst: list[float | None] = []
    for E1, E2 in combinations(scores, 2):
        # scoresはソートされている前提なので、E1 < E2
        # ここでは局所熱平衡状態を考えているので、占有密度はボルツマン分布に従う。ただしボルツマン分布にかかる定数はわからない
        I_lst.append((E2 - E1) ** 3 * alpha * np.exp(-E2 / (Te)))
        dE_lst.append(E2 - E1)
    return dE_lst, I_lst
