from __future__ import annotations
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# dtypeを四倍精度に変える(すっごいメモリ食いそう)


class State(object):
    n = 0

    def __init__(self, v: list[np.int64], score: np.int64):
        self.v = v
        self.score = score
        State.n = len(v)

    def __repr__(self) -> str:
        return str(self.score) + "ε: " + str(self.v)

    def __eq__(self, state: State) -> bool:
        return self.score == state.score

    def __lt__(self, state: State) -> bool:
        return self.score < state.score

    def __le__(self, state: State) -> bool:
        return self.score <= state.score

    def __gt__(self, state: State) -> bool:
        return self.score > state.score

    def __ge__(self, state: State) -> bool:
        return self.score >= state.score

    def __ne__(self, state: State) -> bool:
        return self.score != state.score


class Fermi(object):
    def __init__(self, states: list[State], stable: bool = True, kTe: float = 0.5, ne: float = 1e19):
        self.states = states
        self.n = State.n
        self.ne = ne
        self.kTe = kTe
        self.num_states = len(states)
        self.stable = stable
        self.adj_matrix = np.zeros((self.num_states, self.num_states))
        self.excitation_matrix = np.zeros_like(self.adj_matrix)
        self.deexcitation_matrix = np.zeros_like(self.adj_matrix)
        self.emission_matrix = np.zeros_like(self.adj_matrix)

    # fermi.cppのis_connected()は片方向の遷移が可能かどうかを見てるが、下記のは両方向の遷移が可能かも見れる
    @staticmethod
    def is_connected(s1: State, s2: State) -> bool:
        if len(set([*s1.v, *s2.v])) == s1.n + 1:
            return True
        return False

    # 隣接行列を求める(使わない)
    def make_adj_matrix(self, symm: bool = False) -> None:
        for i in range(self.num_states):
            for j in range(i + 1, self.num_states):
                if Fermi.is_connected(self.states[i], self.states[j]):
                    self.adj_matrix[i, j] = 1
        if symm:
            self.adj_matrix += self.adj_matrix.T

    # 上三角行列
    def _make_matrices(self) -> None:
        for i in range(self.num_states):
            for j in range(i + 1, self.num_states):
                if Fermi.is_connected(self.states[i], self.states[j]):
                    # i→jの遷移
                    self.excitation_matrix[i, j] = self.ne * np.exp(-(self.states[j].score - self.states[i].score) / self.kTe)
                    # j→iの遷移
                    self.deexcitation_matrix[i, j] = self.ne
                    if not self.stable:
                        # j→iの遷移
                        self.emission_matrix[i, j] = (self.states[j].score - self.states[i].score) ** 3

    def _solve_equation(self) -> NDArray[np.float64]:
        if np.all(self.excitation_matrix == 0):
            self._make_matrices()
        C_ = np.diag(self.excitation_matrix.sum(axis=1))
        F_ = np.diag(self.deexcitation_matrix.sum(axis=0))
        C = self.excitation_matrix
        F = self.deexcitation_matrix
        coeff_matrix = C_ - F - C.T + F_
        if not self.stable:
            A_ = np.diag(self.emission_matrix.sum(axis=0))
            A = self.emission_matrix
            coeff_matrix += A_ - A
        self.coeff_matrix = coeff_matrix

        # 対角成分の最大値で正規化し、正負を反転させることで、対角行列のみ負の行列を作成。さらにそこに単位行列を足すことで正行列を作成。
        # ペロン=フロベニウスの定理を用いて、最大の固有値1に対応する固有ベクトルはすべて正の成分を持つことになる。
        normalized_matrix = -coeff_matrix / np.max(np.abs(np.diag(coeff_matrix))) + np.eye(C.shape[0])
        x = Fermi.power_method(normalized_matrix)
        return x / np.sum(x)

    # べき乗法
    @staticmethod
    def power_method(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        # 初期化
        x = np.zeros(matrix.shape[0])
        x[0] = 1
        while True:
            y = np.dot(matrix, x)
            eigen = np.dot(y, x)
            if np.abs(1 - eigen) < 1e-8:
                return y
            x = y / np.linalg.norm(y)

    # score[ε]: 割合 の形の辞書型がほしい
    def get_population(self) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        x = self._solve_equation()
        dct = {}
        for i, state in enumerate(self.states):
            if dct.get(state.score):
                dct[state.score] += x[i]
            else:
                dct[state.score] = x[i]
        scores = np.fromiter(dct.keys(), dtype=int)
        population = np.fromiter(dct.values(), dtype=float)
        return scores, population


def csv_to_states(path: str = "./output/states3.csv") -> list[State]:
    data = pd.read_csv(path, header=0).values
    scores = data[:, 0]
    array = data[:, 1:]
    size = array.shape[0]
    return [State(array[i], scores[i]) for i in range(size)]


def plot(scores, population, stable, kTe, ne, type="plot"):
    if type == "plot":
        plt.plot(scores, population)
    elif type == "scatter":
        plt.scatter(scores, population)
    else:
        plt.plot(scores, population)
    plt.title(f"stable = {stable},   k * T_e = {kTe},   ne = {ne}")
    plt.yscale("log")
    plt.ylim(1e-20, 5)
    plt.xlabel("total energy [ε]")
    plt.ylabel("population [%] (log scale)")
    plt.show()


def plots_dist(
    ne_lst: list[float],
    stable: bool = False,
    kTe: float = 0.5,
    xlim: tuple[float] = None,
    ylim: tuple[float] = None,
    yscale: str = "log",
    figsize: tuple[float] = None,
) -> None:
    states3 = csv_to_states()
    if figsize is not None:
        plt.figure(figsize=figsize)
    for ne in tqdm(ne_lst):
        fermi = Fermi(states3, stable=stable, kTe=kTe, ne=ne)
        scores, population = fermi.get_population()
        plt.plot(scores, population, label=f"ne = {ne}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.title(f"k * T_e = {kTe}")
    plt.yscale(yscale)
    plt.xlabel("total energy [ε]")
    plt.ylabel("population [%] (log scale)")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def show_adj_matrix(fermi: Fermi, figsize: tuple[int] = (10, 10)) -> None:
    fermi.make_adj_matrix(symm=True)
    adj_mat = fermi.adj_matrix
    plt.figure(figsize=figsize)
    plt.pcolormesh(adj_mat, cmap="copper")
    plt.ylim(adj_mat.shape[0] - 1, 0)
