from __future__ import annotations
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os


class State(object):

    n = None

    def __init__(self, v: list[np.int64], score: np.int64):
        self.v = v
        self.score = score

    # HACK: 各インスタンスがメソッドへの参照を持つので効率が悪い。静的メソッドにしてもいい
    @property
    def diff_from_ground(self):
        ground = [i for i in range(State.n)]
        return len(set([*ground, *self.v])) - State.n

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


alpha = 1 / 137  # []
a_0 = 0.52917721 * 1e-10  # [m]
c = 299792458  # [m/s]
E_H = 27.2  # [eV]
e = 1.60217662 * 1e-19  # [C]
pi = np.pi


class SpinFermi(object):
    influx_labels: list[str] = ["excitation from ground level", "excitation from other than ground level", "de-excitation", "emission"]
    outflux_labels: list[str] = ["excitation", "de-excitation", "emission"]

    def __init__(
        self,
        states: list[State],
        Te: float = 0.5,
        ne: float = 1e19,
        eV2eps: float = 1,  # [ε]/[eV]
        scaled_S_0: float = 0.01,
        threshold: float = 0.01,
        loop_lim: int = 500000,
        loop_assure: bool = False,
        gpu: bool = False,
        one_particle_transit: bool = True,
    ):
        self.states = states
        self.ne = ne
        self.Te = Te
        self.eV2eps = eV2eps
        self.scaled_S_0 = scaled_S_0
        self.num_states = len(states)
        self.adj = np.zeros((self.num_states, self.num_states))
        self.excitation = np.zeros_like(self.adj)
        self.deexcitation = np.zeros_like(self.adj)
        self.emission = np.zeros_like(self.adj)
        self.threshold = threshold
        self.loop_lim = loop_lim
        self.loop_assure = loop_assure
        self.gpu = gpu
        self.one_particle_transit = one_particle_transit
        # 電子数10, ne=0.0001, Te=0.5のとき、thresholdを1e-10とすると、power_methodの計算時間は2m36s

    @staticmethod
    def is_connected(s1: State, s2: State) -> bool:
        """
        2つのStateが遷移可能かどうかを判定する
        """
        # fermi.cppのis_connected()は片方向の遷移が可能かどうかを見てるが、下記のは両方向の遷移が可能かも見れる
        if s1.score != s2.score and len(set([*s1.v, *s2.v])) == s1.n + 1:
            return True
        return False

    @staticmethod
    def power_method(
        matrix: NDArray[np.float64], eigen: float, threshold: float, loop_lim: int = 5000000, loop_assure: bool = False, gpu: bool = False
    ) -> NDArray[np.float64]:
        """
        べき乗法により、最大の固有値に対応する固有ベクトルを求める
        """
        if not gpu:
            # 初期化
            x = np.zeros(matrix.shape[0])
            x[:2] = 1 / 2 ** 0.5
            # 最低1,000回はループさせる
            if loop_assure:
                for _ in range(1000):
                    y = np.dot(matrix, x)
                    cur_eigen = np.dot(y, y) / np.dot(y, x)
                    x = y / np.linalg.norm(y)
            for cnt in range(loop_lim):
                y = np.dot(matrix, x)
                cur_eigen = np.dot(y, y) / np.dot(y, x)
                if np.abs(eigen - cur_eigen) < threshold:
                    print(f"loop: {cnt}, diff: {eigen - cur_eigen}, eigen: {eigen}")
                    return x
                x = y / np.linalg.norm(y)
            print(f"loop: {cnt}, diff: {eigen - cur_eigen}, eigen: {eigen}")
            return x
        else:
            import cupy as cp

            matrix = cp.asarray(matrix)
            # 初期化
            x = cp.zeros(matrix.shape[0])
            # x[0] = 1
            x[:2] = 1 / 2 ** 0.5
            # 最低1,000回はループさせる
            if loop_assure:
                for _ in range(1000):
                    y = cp.dot(matrix, x)
                    cur_eigen = cp.dot(y, y) / cp.dot(y, x)
                    x = y / cp.linalg.norm(y)
            for cnt in range(loop_lim):
                y = cp.dot(matrix, x)
                cur_eigen = cp.dot(y, y) / cp.dot(y, x)
                if cp.abs(eigen - cur_eigen) < threshold:
                    print(f"loop: {cnt}, diff: {eigen - cur_eigen}, eigen: {eigen}")
                    return cp.asnumpy(x)
                x = y / cp.linalg.norm(y)
            print(f"loop: {cnt}, diff: {eigen - cur_eigen}, eigen: {eigen}")
            return cp.asnumpy(x)

    @staticmethod
    def get_scores(states: list[State]) -> list[int]:
        """重複なしの各エネルギーエネルギー準位を返す"""
        scores_set = set([state.score for state in states])
        return sorted(list(scores_set))

    @staticmethod
    def get_adj_matrix(states):
        """
        隣接行列を求める
        """
        num_states = len(states)
        adj = np.zeros((num_states, num_states))
        for i in range(num_states):
            for j in range(i + 1, num_states):
                if SpinFermi.is_connected(states[i], states[j]):
                    adj[i, j] = 1
        return adj

    def make_adj_matrix(self, sym: bool = False) -> None:
        """
        隣接行列を求める
        """
        if self.one_particle_transit:
            for i in range(self.num_states):
                for j in range(i + 1, self.num_states):
                    if SpinFermi.is_connected(self.states[i], self.states[j]):
                        self.adj[i, j] = 1
        else:
            for i in range(self.num_states):
                for j in range(i + 1, self.num_states):
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
        beta_ = 2 ** 2.5 / 3 * pi ** 0.5 * alpha * a_0 ** 2 * c * E_H ** 0.5
        gamma_ = 4 / 3 * alpha ** 4 * c / a_0 / (E_H * self.eV2eps) ** 3

        if self.one_particle_transit:
            for i in range(self.num_states):
                for j in range(i + 1, self.num_states):
                    if SpinFermi.is_connected(self.states[i], self.states[j]):
                        # i→jの遷移
                        self.excitation[i, j] = (
                            self.ne / (self.Te) ** 0.5 * self.scaled_S_0 * beta_ * np.exp(-(self.states[j].score - self.states[i].score) / self.Te)
                        )
                        # j→iの遷移
                        self.deexcitation[i, j] = self.ne / (self.Te) ** 0.5 * self.scaled_S_0 * beta_
                        # j→iの遷移
                        self.emission[i, j] = self.scaled_S_0 * gamma_ * (self.states[j].score - self.states[i].score) ** 3
        else:
            for i in range(self.num_states):
                for j in range(i + 1, self.num_states):
                    # i→jの遷移
                    self.excitation[i, j] = (
                        self.ne / (self.Te) ** 0.5 * self.scaled_S_0 * beta_ * np.exp(-(self.states[j].score - self.states[i].score) / self.Te)
                    )
                    # j→iの遷移
                    self.deexcitation[i, j] = self.ne / (self.Te) ** 0.5 * self.scaled_S_0 * beta_
                    # j→iの遷移
                    self.emission[i, j] = self.scaled_S_0 * gamma_ * (self.states[j].score - self.states[i].score) ** 3

    def _solve_equation(self) -> NDArray[np.float64]:
        """
        Xn = 0 の連立方程式を固有値問題とみなし、ペロン=フロベニウスの定理を利用して解を求める。
        事前に係数行列Xを対角成分の最大値で正規化するのではなく、それに足し合わせる単位行列にXの対角成分の最大値をかけることにより、正規化した行列の要素のオーダーが非常に小さくなることを防止し、計算誤差を軽減
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
        A_ = np.diag(sum_along_axis(self.emission, 0))
        A = self.emission
        coeff += A_ - A
        self.coeff = coeff

        # Xの正負を反転させ、対角行列のみ負の行列を作成。そこにXの対角成分の最大値 σ をかけた単位行列を足すことで非負行列を作成
        # ペロン=フロベニウスの定理を用いて、最大の固有値 σ に対応する固有ベクトルはすべて正の成分を持つことになる。
        eigen = np.max(np.abs(np.diag(coeff)))
        non_negative_matrix = -coeff + eigen * np.eye(C.shape[0])
        x = SpinFermi.power_method(
            matrix=non_negative_matrix, eigen=eigen, threshold=self.threshold, loop_lim=self.loop_lim, loop_assure=self.loop_assure, gpu=self.gpu
        )
        return x / np.sum(x)

    def calc_population(self) -> tuple[list[int], NDArray[np.float64]]:
        """
        「縮退状態を分けて考えた状態」ごとのscoreに対する存在割合を計算する。
        """
        population = self._solve_equation()
        scores_per_state = [state.score for state in self.states]
        return scores_per_state, population


def csv_to_states_from_filename(filename: str = "states3_10_spin.csv") -> list[State]:
    """
    各準位における、総エネルギーと電子の位置が記載されたcsvファイルを読み込み、各準位をStateインスタンスとして生成し、リスト化する。
    """
    path = os.path.join(".", "output", filename)
    if not os.path.exists(path):
        path = os.path.join("..", "output", filename)
    data = pd.read_csv(path, header=0).values
    scores = data[:, 0]
    configurations = data[:, 1:]
    # 電子数の設定
    State.n = len(configurations[0])
    return [State(config, score) for config, score in zip(configurations, scores)]
