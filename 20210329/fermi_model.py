from __future__ import annotations
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


class State(object):
    n = None

    def __init__(self, v: list[np.int64], score: np.int64):
        self.v = v
        self.score = score

    def __repr__(self) -> str:
        # return str(self.score) + "ε: " + str(self.v)
        return f"State({self.score}ε: {self.v})"

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
    def __init__(self, states: list[State], equ: bool = True, Te: float = 0.5, ne: float = 1e19):
        self.states = states
        self.ne = ne
        self.Te = Te
        self.num_states = len(states)
        self.equ = equ
        self.adj = np.zeros((self.num_states, self.num_states))
        self.excitation = np.zeros_like(self.adj)
        self.deexcitation = np.zeros_like(self.adj)
        self.emission = np.zeros_like(self.adj)

    @staticmethod
    def is_connected(s1: State, s2: State) -> bool:
        """
        2つのStateが遷移可能かどうかを判定する
        """
        # fermi.cppのis_connected()は片方向の遷移が可能かどうかを見てるが、下記のは両方向の遷移が可能かも見れる
        if len(set([*s1.v, *s2.v])) == s1.n + 1:
            return True
        return False

    @staticmethod
    def power_method(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        べき乗法により、最大の固有値に対応する固有ベクトルを求める
        """
        # 初期化
        x = np.zeros(matrix.shape[0])
        x[0] = 1
        eigen_past = 0
        while True:
            y = np.dot(matrix, x)
            eigen = np.dot(y, y) / np.dot(y, x)
            # ne=0.001のとき1minかかった。eigen = 0.9999999999464453なので精度は低い
            if np.abs(eigen_past - eigen) < 1e-15:
                return x
            x = y / np.linalg.norm(y)
            eigen_past = eigen

    def make_adj_matrix(self, sym: bool = False) -> None:
        """
        隣接行列を求める
        """
        for i in range(self.num_states):
            for j in range(i + 1, self.num_states):
                if Fermi.is_connected(self.states[i], self.states[j]):
                    self.adj[i, j] = 1
        if sym:
            self.adj += self.adj.T

    def show_adj_matrix(self, figsize: tuple[int] = (10, 10)) -> None:
        """
        隣接行列を求めて表示する
        """
        self.make_adj_matrix(sym=True)
        plt.figure(figsize=figsize)
        plt.pcolormesh(self.adj, cmap="copper")
        plt.ylim(self.adj.shape[0] - 1, 0)
        plt.show()

    def _make_matrices(self) -> None:
        """
        対角成分が0の各種上三角行列を求める
        """
        for i in range(self.num_states):
            for j in range(i + 1, self.num_states):
                if Fermi.is_connected(self.states[i], self.states[j]):
                    # i→jの遷移
                    self.excitation[i, j] = self.ne * np.exp(-(self.states[j].score - self.states[i].score) / self.Te)
                    # j→iの遷移
                    self.deexcitation[i, j] = self.ne
                    if not self.equ:
                        # j→iの遷移
                        self.emission[i, j] = (self.states[j].score - self.states[i].score) ** 3

    def _solve_equation(self, use_power: bool = False) -> NDArray[np.float64]:
        """
        Xn = 0 の連立方程式を固有値問題とみなし、ペロン=フロベニウスの定理を利用して解を求める
        """

        # ndarray.sum(axis=0)では誤差が出てしまうので、その代用
        def sum_along_axis(matrix: NDArray[np.float64], axis: int = 0):
            return np.apply_along_axis(np.sum, axis, matrix)

        if np.all(self.excitation == 0):
            self._make_matrices()
        C_ = np.diag(sum_along_axis(self.excitation, 1))
        F_ = np.diag(sum_along_axis(self.deexcitation, 0))
        C = self.excitation
        F = self.deexcitation
        coeff = C_ - F - C.T + F_
        if not self.equ:
            A_ = np.diag(sum_along_axis(self.emission, 0))
            A = self.emission
            coeff += A_ - A
        self.coeff = coeff

        # 対角成分の最大値で正規化し、正負を反転させることで、対角行列のみ負の行列を作成。さらにそこに単位行列を足すことで正行列を作成。
        # ペロン=フロベニウスの定理を用いて、最大の固有値1に対応する固有ベクトルはすべて正の成分を持つことになる。
        normalized = -coeff / np.max(np.abs(np.diag(coeff))) + np.eye(C.shape[0])
        if use_power:
            x = Fermi.power_method(normalized)
        else:
            eigs, xs = np.linalg.eig(normalized)
            x = np.abs(xs[:, np.argmax(eigs)])
        return x / np.sum(x)

    def get_distribution(self, use_power: bool = False) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        scoreに対する、縮退状態について和をとった確率密度分布を計算する
        """
        n = self._solve_equation(use_power)
        dct = {}
        for rate, state in zip(n, self.states):
            if dct.get(state.score):
                dct[state.score] += rate
            else:
                dct[state.score] = rate
        scores = np.fromiter(dct.keys(), dtype=int)
        distribution = np.fromiter(dct.values(), dtype=float)
        return scores, distribution

    def get_population(self, use_power: bool = False) -> tuple[list[int], NDArray[np.float64]]:
        """
        「縮退状態を分けて考えた状態」ごとのscoreに対する存在割合を計算する。
        """
        population = self._solve_equation(use_power)
        scores = [state.score for state in self.states]
        return scores, population

    def get_mean_distribution(self, use_power: bool = False) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        scoreに対する、縮退状態について和をとり、縮退度で平均した確率密度分布を計算する
        """
        n = self._solve_equation(use_power)
        present_score = 0
        dct = {}
        degeneracy = 0
        for rate, state in zip(n, self.states):
            if state.score != present_score:
                # １つ前のscoreの割合に対して、縮退度で平均をとる
                if degeneracy != 0:
                    dct[state.score - 1] /= degeneracy

                dct[state.score] = rate
                present_score = state.score
                degeneracy = 1
            else:
                dct[state.score] += rate
                degeneracy += 1
        dct[self.states[-1].score] /= degeneracy
        scores = np.fromiter(dct.keys(), dtype=int)
        mean_distribution = np.fromiter(dct.values(), dtype=float)
        return scores, mean_distribution


def csv_to_states(path: str = "./output/states3.csv") -> list[State]:
    """
    各準位における、総エネルギーと電子の位置が記載されたcsvファイルを読み込み、各準位をStateインスタンスとして生成し、リスト化する。
    """
    data = pd.read_csv(path, header=0).values
    scores = data[:, 0]
    configurations = data[:, 1:]
    # 電子数の設定
    State.n = len(configurations[0])
    return [State(config, score) for config, score in zip(configurations, scores)]


# 使わない
def plot(scores, population, equ, Te, ne, type="plot"):
    if type == "plot":
        plt.plot(scores, population)
    elif type == "scatter":
        plt.scatter(scores, population)
    else:
        plt.plot(scores, population)
    plt.title(f"equilibrium = {equ},   T_e = {Te},   ne = {ne}")
    plt.yscale("log")
    plt.ylim(1e-20, 5)
    plt.xlabel("total energy [ε]")
    plt.ylabel("population [%] (log scale)")
    plt.show()


def plots_dist(
    ne_lst: list[float],
    Te: float = 0.5,
    include_equ: bool = False,
    use_power: bool = False,
    xlim: tuple[float] = None,
    ylim: tuple[float] = None,
    yscale: str = "log",
    figsize: tuple[float] = None,
) -> None:
    """
    各neの値におけるフェルミガスモデルを構築し、総エネルギーを横軸にとり、縮退状態について和をとった占有密度を縦軸logスケールでプロットする
    """
    states3 = csv_to_states()
    if figsize is not None:
        plt.figure(figsize=figsize)
    if include_equ:
        fermi = Fermi(states3, equ=True, Te=Te, ne=1e19)
        scores, population = fermi.get_distribution(use_power)
        plt.plot(scores, population, label="equilibrium", marker=".", linewidth=0.8, ms=3)
    for ne in tqdm(ne_lst):
        fermi = Fermi(states3, equ=False, Te=Te, ne=ne)
        scores, population = fermi.get_distribution(use_power)
        plt.plot(scores, population, label=f"ne = {ne}", marker=".", linewidth=0.8, ms=3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.title(f"distribution (T_e = {Te})")
    plt.yscale(yscale)
    plt.xlabel("E (total energy) [ε]")
    plt.ylabel("P(E) (log scale)")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def plots_mean_dist(
    ne_lst: list[float],
    Te: float = 0.5,
    include_equ: bool = False,
    use_power: bool = False,
    xlim: tuple[float] = None,
    ylim: tuple[float] = None,
    yscale: str = "log",
    figsize: tuple[float] = None,
) -> None:
    """
    各neの値におけるフェルミガスモデルを構築し、総エネルギーを横軸、縮退状態について和をとり縮退度で平均した占有密度を縦軸logスケールでプロットする
    """
    states3 = csv_to_states()
    if figsize is not None:
        plt.figure(figsize=figsize)
    if include_equ:
        fermi = Fermi(states3, equ=True, Te=Te, ne=1e19)
        scores, population = fermi.get_mean_distribution(use_power)
        plt.plot(scores, population, label="equilibrium", marker=".", linewidth=0.8, ms=3)
    for ne in tqdm(ne_lst):
        fermi = Fermi(states3, equ=False, Te=Te, ne=ne)
        scores, population = fermi.get_mean_distribution(use_power)
        plt.plot(scores, population, label=f"ne = {ne}", marker=".", linewidth=0.8, ms=3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.title(f"mean distribution (T_e = {Te})")
    plt.yscale(yscale)
    plt.xlabel("E (total energy) [ε]")
    plt.ylabel("P(E)/ρ(E) (log scale)")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def plots_poplulation(
    ne_lst: list[float],
    Te: float = 0.5,
    include_equ: bool = False,
    use_power: bool = False,
    xlim: tuple[float] = None,
    ylim: tuple[float] = None,
    yscale: str = "log",
    figsize: tuple[float] = None,
) -> None:
    """
    各neの値におけるフェルミガスモデルを構築し、総エネルギーを横軸にとり、縮退状態を別々で考えた各状態の占有密度を縦軸logスケールでプロットする
    """
    states3 = csv_to_states()
    if figsize is not None:
        plt.figure(figsize=figsize)
    if include_equ:
        fermi = Fermi(states3, equ=True, Te=Te, ne=1e19)
        scores, population = fermi.get_population(use_power)
        plt.scatter(scores, population, label="equilibrium", s=2)
    for ne in tqdm(ne_lst):
        fermi = Fermi(states3, equ=False, Te=Te, ne=ne)
        scores, population = fermi.get_population(use_power)
        plt.scatter(scores, population, label=f"ne = {ne}", s=2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.title(f"population (T_e = {Te})")
    plt.yscale(yscale)
    plt.xlabel("E (total energy) [ε]")
    plt.ylabel("population [%] (log scale)")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()
