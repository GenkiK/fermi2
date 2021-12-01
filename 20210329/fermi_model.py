from __future__ import annotations
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["font.size"] = 12


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


class Fermi(object):
    influx_labels: list[str] = ["基底状態からの衝突励起", "基底状態以外からの衝突励起", "衝突脱励起", "放射脱励起"]
    outflux_labels: list[str] = ["衝突励起", "衝突脱励起", "放射脱励起"]

    def __init__(self, states: list[State], equ: bool = True, Te: float = 0.5, ne: float = 1e19, threshold: float = 1e-15):
        self.states = states
        self.ne = ne
        self.Te = Te
        self.num_states = len(states)
        self.equ = equ
        self.adj = np.zeros((self.num_states, self.num_states))
        self.excitation = np.zeros_like(self.adj)
        self.deexcitation = np.zeros_like(self.adj)
        self.emission = np.zeros_like(self.adj)
        self.threshold = threshold
        # 電子数10, ne=0.0001, Te=0.5のとき、thresholdを1e-10とすると、power_methodの計算時間は2m36s

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
    def power_method(matrix: NDArray[np.float64], threshold: float = 1e-15) -> NDArray[np.float64]:
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
            if np.abs(eigen_past - eigen) < threshold:
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

    # def _solve_equation(self, use_power: bool = True) -> NDArray[np.float64]:
    #     """
    #     Xn = 0 の連立方程式を固有値問題とみなし、ペロン=フロベニウスの定理を利用して解を求める
    #     """

    #     # ndarray.sum(axis=0)では誤差が出てしまうので、その代用
    #     def sum_along_axis(matrix: NDArray[np.float64], axis: int = 0):
    #         return np.apply_along_axis(np.sum, axis, matrix)

    #     if np.all(self.excitation == 0):
    #         self._make_matrices()
    #     C_ = np.diag(sum_along_axis(self.excitation, 1))
    #     F_ = np.diag(sum_along_axis(self.deexcitation, 0))
    #     C = self.excitation
    #     F = self.deexcitation
    #     coeff = C_ - F - C.T + F_
    #     if not self.equ:
    #         A_ = np.diag(sum_along_axis(self.emission, 0))
    #         A = self.emission
    #         coeff += A_ - A
    #     self.coeff = coeff

    #     # 対角成分の最大値で正規化し、正負を反転させることで、対角行列のみ負の行列を作成。さらにそこに単位行列を足すことで正行列を作成。
    #     # ペロン=フロベニウスの定理を用いて、最大の固有値1に対応する固有ベクトルはすべて正の成分を持つことになる。
    #     normalized = -coeff / np.max(np.abs(np.diag(coeff))) + np.eye(C.shape[0])
    #     if use_power:
    #         x = Fermi.power_method(normalized)
    #     else:
    #         eigs, xs = np.linalg.eig(normalized)
    #         x = np.abs(xs[:, np.argmax(eigs)])
    #     return x / np.sum(x)

    def _solve_equation(self, use_power: bool = True) -> NDArray[np.float64]:
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
        if not self.equ:
            A_ = np.diag(sum_along_axis(self.emission, 0))
            A = self.emission
            coeff += A_ - A
        self.coeff = coeff

        # Xの正負を反転させ、対角行列のみ負の行列を作成。そこにXの対角成分の最大値 σ をかけた単位行列を足すことで非負行列を作成
        # ペロン=フロベニウスの定理を用いて、最大の固有値 σ に対応する固有ベクトルはすべて正の成分を持つことになる。
        non_negative_matrix = -coeff + np.max(np.abs(np.diag(coeff))) * np.eye(C.shape[0])
        if use_power:
            x = Fermi.power_method(non_negative_matrix, self.threshold)
        else:
            eigs, xs = np.linalg.eig(non_negative_matrix)
            x = np.abs(xs[:, np.argmax(eigs)])
        return x / np.sum(x)

    def calc_distribution(self, use_power: bool = True) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
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

    def calc_population(self, use_power: bool = True) -> tuple[list[int], NDArray[np.float64]]:
        """
        「縮退状態を分けて考えた状態」ごとのscoreに対する存在割合を計算する。
        """
        population = self._solve_equation(use_power)
        scores_per_state = [state.score for state in self.states]
        return scores_per_state, population

    def calc_mean_distribution(self, use_power: bool = True) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
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

    def calc_percentage_fluxes(self) -> tuple[dict[int : list[float]]]:
        """
        Returns:
            percentage_influx_dict, percentage_outflux_dict

        Details:
            percentage_influx_dict: {score: [percentage_c_ground, percentage_c, percentage_f, percentage_a]}
            percentage_outflux_dict: {score: [percentage_c, percentage_f, percentage_a]}

        Examples:
            percentage_influx_dict, percentage_outflux_dict = fermi.calc_percentage_fluxes()
            percentage_influxes = np.array(list(percentage_influx_dict.values())).T
            plt.stackplot(list(percentage_influx_dict.keys()), *list(percentage_influxes), labels=Fermi.influx_labels)
            plt.xticks(list(percentage_influx_dict.keys()))
            plt.show()
        """
        scores, population = self.calc_population()
        C_flux = np.dot(np.diag(population), self.excitation)
        F_flux = self.deexcitation * population
        A_flux = self.emission * population

        # エネルギー準位ごとにfluxをまとめて、各エネルギー準位におけるfluxの割合を求める関数
        def _calc_percentage_flux_dict(scores, C, F, A: list[NDArray[np.float64]], influx: bool) -> dict[int : list[float]]:
            C_ground = None
            if influx:
                C_ground = C[0]
                C = C[1:]
            C = np.sum(C, axis=1 ^ int(influx))
            F = np.sum(F, axis=int(influx))
            A = np.sum(A, axis=int(influx))

            # flux_dict = {score: [(percentage_c_ground,) percentage_c, percentage_f, percentage_a]}
            flux_dict = {}
            previous_score = scores[0]
            c_acc = 0  # acc means accumulator
            f_acc = 0
            a_acc = 0

            if influx:
                # initialize c_ground_acc
                c_ground_acc = 0
                for score, c_ground, c, f, a in zip(scores, C_ground, C, F, A):
                    if score == previous_score:
                        c_ground_acc += c_ground
                        c_acc += c
                        f_acc += f
                        a_acc += a
                    else:
                        total = c_ground_acc + c_acc + f_acc + a_acc
                        flux_dict[previous_score] = [c_ground_acc, c_acc, f_acc, a_acc] / total
                        previous_score = score
                        c_ground_acc = c_ground
                        c_acc = c
                        f_acc = f
                        a_acc = a
                total = c_ground_acc + c_acc + f_acc + a_acc
                flux_dict[previous_score] = [c_ground_acc, c_acc, f_acc, a_acc] / total
                return flux_dict

            else:
                for score, c, f, a in zip(scores, C, F, A):
                    if score == previous_score:
                        c_acc += c
                        f_acc += f
                        a_acc += a
                    else:
                        total = c_acc + f_acc + a_acc
                        flux_dict[previous_score] = [c_acc, f_acc, a_acc] / total
                        previous_score = score
                        c_acc = c
                        f_acc = f
                        a_acc = a
                total = c_acc + f_acc + a_acc
                flux_dict[previous_score] = [c_acc, f_acc, a_acc] / total
                return flux_dict

        percentage_influx_dict = _calc_percentage_flux_dict(scores, C_flux, F_flux, A_flux, influx=True)
        percentage_outflux_dict = _calc_percentage_flux_dict(scores, C_flux, F_flux, A_flux, influx=False)
        return percentage_influx_dict, percentage_outflux_dict

    # scoreをキーとする辞書型で返すから、冗長性は低いけど使い勝手が悪い。ほぼ使わない
    def calc_percentage_influx_per_state_dict(self) -> dict[float, list[dict[float, list[float]]]]:
        """
        Returns:
            percentage_influx_per_state_dict

        Details:
            percentage_influx_per_state_dict: dict[float, list[dict[float, list[float]]]]
                = {score: [{population: [percentage_c_ground, percentage_c, percentage_f, percentage_a]}]}
        """
        scores, population = self.calc_population()
        C_flux = np.dot(np.diag(population), self.excitation)
        F_flux = self.deexcitation * population
        A_flux = self.emission * population

        C_influx_ground = C_flux[0]
        C_influx = np.sum(C_flux[1:], axis=0)
        F_influx = np.sum(F_flux, axis=1)
        A_influx = np.sum(A_flux, axis=1)

        percentage_influx_per_state_dict: dict[float, list[dict[float, list[float]]]] = {}
        for i in range(len(scores)):
            score = scores[i]
            n = population[i]
            c_ground = C_influx_ground[i]
            c = C_influx[i]
            f = F_influx[i]
            a = A_influx[i]
            total = c_ground + c + f + a
            percentage_lst = [elem / total for elem in [c_ground, c, f, a]]
            if percentage_influx_per_state_dict.get(score):
                percentage_influx_per_state_dict[score].append({n: percentage_lst})
            else:
                percentage_influx_per_state_dict[score] = [{n: percentage_lst}]

        return percentage_influx_per_state_dict

    def calc_percentage_influx_per_state(self) -> tuple[list[float], list[float], list[list[float]]]:
        """
        Returns:
            scores_per_state, population, percentage_influx_per_state

        Details:
            percentage_influx_per_state = [[percentage_c_ground, percentage_c, percentage_f, percentage_a],...]
        """
        scores_per_state, population = self.calc_population()
        C_flux = np.dot(np.diag(population), self.excitation)
        F_flux = self.deexcitation * population
        A_flux = self.emission * population

        C_influx_ground = C_flux[0]
        C_influx = np.sum(C_flux[1:], axis=0)
        F_influx = np.sum(F_flux, axis=1)
        A_influx = np.sum(A_flux, axis=1)

        percentage_influx_per_state = []
        for i in range(len(scores_per_state)):
            c_ground = C_influx_ground[i]
            c = C_influx[i]
            f = F_influx[i]
            a = A_influx[i]
            total = c_ground + c + f + a
            percentage_lst = [elem / total for elem in [c_ground, c, f, a]]
            percentage_influx_per_state.append(percentage_lst)

        return scores_per_state, population, percentage_influx_per_state

    def calc_population_per_diff(self) -> tuple[tuple[list[int], list[float]]]:
        """
        Returns:
            (scores_per_state_1, population_1), (scores_per_state_2, population_2), (scores_per_state_3, population_3)
        """
        scores_per_state_1 = []
        scores_per_state_2 = []
        scores_per_state_3 = []

        population_1 = []
        population_2 = []
        population_3 = []
        _, population = self.calc_population()
        for state, n in zip(self.states, population):
            if state.diff_from_ground == 1:
                scores_per_state_1.append(state.score)
                population_1.append(n)
            elif state.diff_from_ground == 2:
                scores_per_state_2.append(state.score)
                population_2.append(n)
            elif state.diff_from_ground == 3:  # 基底状態を含んでしまうことを防止するため、elseを使わない
                scores_per_state_3.append(state.score)
                population_3.append(n)

        return (scores_per_state_1, population_1), (scores_per_state_2, population_2), (scores_per_state_3, population_3)

    def calc_percentage_influx_per_state_each_diff(self) -> tuple[list[list[int]], list[list[float]], list[list[list[float]]]]:
        """
        各状態におけるinfluxの内訳の割合を、基底状態とのdiffごとに出力する

        Returns:
            list[list[int]]: diffごとに分けた、各状態のscore

            list[list[float]]: diffごとに分けた、各状態のpopulation

            list[list[list[float]]]: diffごとに分けた、各状態におけるinfluxの内訳の割合。diff -> state -> percentage という順で階層分けされている
                (ex): [[[percentage_c_ground, percentage_c, percentage_f, percentage_a],...], []], [[], []]]
        """
        scores_per_state, population = self.calc_population()
        C_flux = np.dot(np.diag(population), self.excitation)
        F_flux = self.deexcitation * population
        A_flux = self.emission * population

        C_influx_ground = C_flux[0]
        C_influx = np.sum(C_flux[1:], axis=0)
        F_influx = np.sum(F_flux, axis=1)
        A_influx = np.sum(A_flux, axis=1)

        # initialize
        percentage_influx_per_state_each_diff = [[], [], []]
        scores_per_state_each_diff = [[], [], []]
        population_each_diff = [[], [], []]

        for i, state in enumerate(self.states):
            # 基底状態のinfluxの内訳は含めないこととする
            if i == 0:
                continue
            c_ground = C_influx_ground[i]
            c = C_influx[i]
            f = F_influx[i]
            a = A_influx[i]
            total = c_ground + c + f + a
            diff_idx = state.diff_from_ground - 1
            percentage_lst = [elem / total for elem in [c_ground, c, f, a]]
            percentage_influx_per_state_each_diff[diff_idx].append(percentage_lst)
            scores_per_state_each_diff[diff_idx].append(scores_per_state[i])
            population_each_diff[diff_idx].append(population[i])
        return scores_per_state_each_diff, population_each_diff, percentage_influx_per_state_each_diff


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


def csv_to_states_from_filename(filename: str = "states3.csv") -> list[State]:
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


def plots_percentage_fluxes(
    ne_lst: list[float],
    Te: float = 0.5,
    figsize: tuple[float] = (13, 4),
) -> None:
    states3 = csv_to_states_from_filename()
    for ne in ne_lst:
        fig = plt.figure(figsize=figsize)
        fermi = Fermi(states3, equ=False, Te=Te, ne=ne)
        percentage_influx_dict, percentage_outflux_dict = fermi.calc_percentage_fluxes()
        percentage_influxes = np.array(list(percentage_influx_dict.values())).T
        subfig1 = fig.add_subplot(1, 2, 1)
        subfig1.stackplot(
            list(percentage_influx_dict.keys()),
            *list(percentage_influxes),
            labels=Fermi.influx_labels,
            colors=["Red", "Orange", "LimeGreen", "DodgerBlue"],
        )
        subfig1.set_xlabel("エネルギー準位 E")
        subfig1.set_ylabel("流入量の割合 [%]")
        subfig1.set_title(f"influx (ne={ne},Te={Te})")
        subfig1.set_xticks(list(percentage_influx_dict.keys()))
        subfig1.set_xmargin(0)
        subfig1.set_ymargin(0)
        subfig1.legend()

        percentage_outfluxes = np.array(list(percentage_outflux_dict.values())).T
        subfig2 = fig.add_subplot(1, 2, 2)
        subfig2.stackplot(
            list(percentage_outflux_dict.keys()),
            *list(percentage_outfluxes),
            labels=Fermi.outflux_labels,
            colors=["Red", "LimeGreen", "DodgerBlue"],
        )
        subfig2.set_xlabel("エネルギー準位 E")
        subfig2.set_ylabel("流出量の割合 [%]")
        subfig2.set_title(f"outflux (ne={ne},Te={Te})")
        subfig2.set_xticks(list(percentage_outflux_dict.keys()))
        subfig2.set_xmargin(0)
        subfig2.set_ymargin(0)
        subfig2.legend()
        plt.show()


def plot_percentage_influx_per_state_3d(
    ne: float,
    Te: float = 0.5,
    is_dark: bool = False,
    diff: list[int] = None,
    separated: bool = False,
    width: float = 0.03,
    figsize: int = 8,
    labelsize: int = 8,
) -> None:
    states3 = csv_to_states_from_filename()
    fermi = Fermi(states3, equ=False, Te=Te, ne=ne)
    title = fr"$n_e={ne}, T_e={Te}$"

    population = []
    scores_per_state = []
    percentage_influx_per_state = []
    is_diff = False

    if separated and diff is not None:
        scores_per_state_each_diff, population_each_diff, percentage_influx_per_state_each_diff = fermi.calc_percentage_influx_per_state_each_diff()
        # 各行の要素数がバラバラなpopulation_each_diffを１次元ndarrayに展開
        flat_population_each_diff = np.array(sum(population_each_diff, []))
        min_log_population = np.min(np.log(flat_population_each_diff))

        for i in range(3):
            if i + 1 in diff:
                scores_per_state.append(scores_per_state_each_diff[i])
                population.append(population_each_diff[i])
                percentage_influx_per_state.append(percentage_influx_per_state_each_diff[i])
        plot_some_bar3d(
            dzs_lst=percentage_influx_per_state,
            x_pos_lst=scores_per_state,
            y_pos_lst=population,
            titles=[title + f", diff=[{i}]" for i in diff],
            labels=Fermi.influx_labels,
            labelsize=labelsize,
            width=width,
            each_figsize=figsize,
            is_dark=is_dark,
            xticks=sorted(list(set([state.score for state in fermi.states]))),
            ylim=(min_log_population - 10, 10),
        )

    else:
        if diff is None:
            scores_per_state, population, percentage_influx_per_state = fermi.calc_percentage_influx_per_state()
            min_log_population = np.min(np.log(population))
        else:
            is_diff = True
            (
                scores_per_state_each_diff,
                population_each_diff,
                percentage_influx_per_state_each_diff,
            ) = fermi.calc_percentage_influx_per_state_each_diff()
            # 各行の要素数がバラバラなpopulation_each_diffを１次元ndarrayに展開
            flat_population_each_diff = np.array(sum(population_each_diff, []))
            min_log_population = np.min(np.log(flat_population_each_diff))

            for i in range(3):
                if i + 1 in diff:
                    scores_per_state += scores_per_state_each_diff[i]
                    population += population_each_diff[i]
                    percentage_influx_per_state += percentage_influx_per_state_each_diff[i]

        plot_bar3d(
            dzs=percentage_influx_per_state,
            x_pos=scores_per_state,
            y_pos=population,
            title=title + f", diff={diff}" if is_diff else title,
            labels=Fermi.influx_labels,
            labelsize=labelsize,
            width=width,
            figsize=figsize,
            is_dark=is_dark,
            xticks=sorted(list(set([state.score for state in fermi.states]))) if is_diff else None,
            ylim=(min_log_population - 10, 10),
        )


def plot_some_bar3d(
    dzs_lst: list[list[list[float]]],
    x_pos_lst: list[list[int]],
    y_pos_lst: list[list[float]],
    titles: list[str],
    labels: list[str],
    labelsize: int,
    width: float,
    each_figsize: int,
    is_dark: bool,
    xticks: list[int] = None,
    ylim: list[float] = None,
):
    """
    Args:
        dzs_lst (list[list[list[float]]])  各グラフのz軸方向に重ねていく２次元配列を格納した配列。２次元配列の１つの行に、ある点(x, y)に重ねていきたいdzの要素を格納したものを渡す。

        x_pos_lst (list[list[int]])  各グラフのxの座標の配列、の配列

        y_pos_lst (list[list[float]])  各グラフのyの座標の配列、の配列

        titles (list[str])  各グラフのタイトルの配列

        labels (list[str])  z軸方向に重ねていくもののラベル

        labelsize (int)  ラベルのサイズ

        width (float)  barの幅

        each_figsize (int)  各グラフの大きさ

        is_dark (bool)  ダークモードでプロットするかどうかを決める真理値

        xticks (list[int])  x軸のticks。指定しなければ自動的にいい感じに決定してくれる。

        ylim (list[float])  y軸の範囲。diffごとにプロットするときなどに指定することで、同じneで実験したときの表示されるy軸の範囲を一定にできる

    Example:
        plot_bar3d(
            dzs=percentage_influx_per_state,
            x_pos=scores_per_state,
            y_pos=population,
            title=fr"$n_e={ne},  T_e={Te}$",
            labels=Fermi.influx_labels,
            is_dark=is_dark,
        )
    """
    plot_num = len(dzs_lst)
    fig = plt.figure(figsize=(each_figsize * plot_num, each_figsize), facecolor="black" if is_dark else "white")
    for i, (dzs, x_pos, y_pos, title) in enumerate(zip(dzs_lst, x_pos_lst, y_pos_lst, titles)):
        ax = fig.add_subplot(1, plot_num, i + 1, projection="3d")
        plot_bar3d(
            dzs=dzs,
            x_pos=x_pos,
            y_pos=y_pos,
            title=title,
            labels=labels,
            labelsize=labelsize,
            width=width,
            figsize=each_figsize,
            is_dark=is_dark,
            xticks=xticks,
            ylim=ylim,
            ax=ax,
            show=False,
        )
    plt.show()


def plot_bar3d(
    dzs: list[list[float]],
    x_pos: list[int],
    y_pos: list[float],
    title: str,
    labels: list[str],
    labelsize: int,
    width: float,
    figsize: int,
    is_dark: bool,
    xticks: list[int] = None,
    ylim: list[float] = None,
    ax: plt.Axes = None,
    show: bool = True,
):
    """
    Args:
        dzs (list[list[float]])  z軸方向に重ねていく２次元配列。１つの行に、ある点(x, y)に重ねていきたいdzの要素を格納したものを渡す。

        x_pos (list[int])  xの座標の配列

        y_pos (list[float])  yの座標の配列

        title (str)  グラフのタイトル

        labels (list[str])  z軸方向に重ねていくもののラベル

        labelsize (int)  ラベルのサイズ

        width (float)  barの幅

        figsize (int)  グラフの大きさ

        is_dark (bool)  ダークモードでプロットするかどうかを決める真理値

        xticks (list[int])  x軸のticks。指定しなければ自動的にいい感じに決定してくれる。

        ylim (list[float])  y軸の範囲。diffごとにプロットするときなどに指定することで、同じneで実験したときの表示されるy軸の範囲を一定にできる

        ax (plt.Axes)  Axesオブジェクト。この関数をoverwrapした、複数個のグラフを横並びでプロットする関数で使用する。

        show (bool)  plt.show()を実行するかどうかを決める真理値。plot_bar3d()をoverwrapする関数内で使用する。

    Example:
        plot_bar3d(
            dzs=percentage_influx_per_state,
            x_pos=scores_per_state,
            y_pos=population,
            title=fr"$n_e={ne},  T_e={Te}$",
            labels=Fermi.influx_labels,
            is_dark=is_dark,
        )
    """
    import matplotlib.ticker as mticker

    def log_tick_formatter(val, pos=None):
        return f"$10^{{{int(val)}}}$"

    def default_formatter(val, pos=None):
        return fr"${val}$"

    y_pos = np.log(np.array(y_pos))
    z_pos = np.zeros(len(x_pos))
    dx = np.full(len(x_pos), width)
    dy = np.full(len(y_pos), np.exp(width))
    dzs = np.array(dzs).T

    if ax is None:
        fig = plt.figure(figsize=(figsize, figsize), facecolor="black" if is_dark else "white")
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(default_formatter))
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(default_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    _z_pos = np.copy(z_pos)
    colors = ["Red", "Gold", "GreenYellow", "DeepSkyBlue"] if is_dark else ["Red", "Gold", "Green", "SkyBlue"]
    for dz, color in zip(dzs, colors):
        ax.bar3d(x_pos, y_pos, _z_pos, dx, dy, dz, alpha=1, color=color, shade=False)
        _z_pos += dz

    color_proxies = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in colors]
    ax.legend(color_proxies, labels, fontsize=labelsize)

    xticks = sorted(list(set(x_pos))) if xticks is None else xticks
    ax.set_xticks([xtick for xtick in xticks if xtick % 2 == 1])
    ax.set_zticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title, color="white" if is_dark else "black")
    ax.set_xlabel("エネルギー準位")
    ax.set_ylabel("占有密度 (log)")
    ax.set_zlabel("流入量の割合")
    ax.grid(False)
    ax.set_facecolor("black" if is_dark else "white")
    ax.xaxis.label.set_color("white" if is_dark else "black")
    ax.yaxis.label.set_color("white" if is_dark else "black")
    ax.zaxis.label.set_color("white" if is_dark else "black")
    ax.tick_params(axis="x", colors="white" if is_dark else "black")
    ax.tick_params(axis="y", colors="white" if is_dark else "black")
    ax.tick_params(axis="z", colors="white" if is_dark else "black")
    if show:
        plt.show()


def plots_dist(
    ne_lst: list[float],
    Te: float = 0.5,
    include_equ: bool = False,
    use_power: bool = True,
    xlim: tuple[float] = None,
    ylim: tuple[float] = None,
    yscale: str = "log",
    figsize: tuple[float] = None,
    labelsize: int = None,
) -> None:
    """
    各neの値におけるフェルミガスモデルを構築し、総エネルギーを横軸にとり、縮退状態について和をとった占有密度を縦軸logスケールでプロットする
    """
    states3 = csv_to_states_from_filename()
    if figsize is not None:
        plt.figure(figsize=figsize)
    if include_equ:
        fermi = Fermi(states3, equ=True, Te=Te, ne=1e19)
        scores, population = fermi.calc_distribution(use_power)
        plt.plot(scores, population, label="equilibrium", marker=".", linewidth=0.8, ms=3)
    for ne in tqdm(ne_lst):
        fermi = Fermi(states3, equ=False, Te=Te, ne=ne)
        scores, population = fermi.calc_distribution(use_power)
        plt.plot(scores, population, label=fr"$n_e$ = {ne}", marker=".", linewidth=0.8, ms=3)
    # plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.legend(loc="lower left", fontsize=labelsize)
    plt.title(fr"distribution ($T_e$ = {Te})")
    plt.yscale(yscale)
    plt.xlabel(r"$E (total energy) [\epsilon]$")
    plt.ylabel(r"$P(E)$  ($\log$ scale)")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def plots_mean_dist(
    ne_lst: list[float],
    Te: float = 0.5,
    include_equ: bool = False,
    use_power: bool = True,
    xlim: tuple[float] = None,
    ylim: tuple[float] = None,
    figsize: tuple[float] = None,
    labelsize: int = None,
    titlesize: int = None,
) -> None:
    """
    各neの値におけるフェルミガスモデルを構築し、総エネルギーを横軸、縮退状態について和をとり縮退度で平均した占有密度を縦軸logスケールでプロットする
    """
    states3 = csv_to_states_from_filename()
    scores = None
    if figsize is not None:
        plt.figure(figsize=figsize)
    if include_equ:
        ne = 1e20
        fermi = Fermi(states3, equ=False, Te=Te, ne=ne)
        scores, distribution = fermi.calc_mean_distribution(use_power)
        plt.plot(scores, distribution, label=fr"$n_e$={ne} (equilibrium)", marker=".", linewidth=1, ms=4)
    for ne in tqdm(ne_lst):
        fermi = Fermi(states3, equ=False, Te=Te, ne=ne)
        scores, distribution = fermi.calc_mean_distribution(use_power)
        plt.plot(scores, distribution, label=fr"$n_e$={ne}", marker=".", linewidth=1, ms=4)
    # plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=labelsize)
    plt.legend(loc="lower left", fontsize=labelsize)
    plt.xticks(scores)
    plt.title(fr"縮退度で平均した占有密度分布 ($T_e$ = {Te})", fontsize=titlesize)
    plt.yscale("log")
    plt.xlabel(r"エネルギー準位 $E$ $[\epsilon]$", fontsize=labelsize)
    plt.ylabel(r"$P(E)$/縮退度  ($\log$ scale)", fontsize=labelsize)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def plots_mean_dist_compare(
    ne_lst: list[float],
    Te: float = 0.5,
    include_equ: bool = False,
    use_power: bool = True,
    xlim: tuple[float] = None,
    ylim: tuple[float] = None,
    figsize: tuple[float] = None,
) -> None:
    states3 = csv_to_states_from_filename()
    if figsize is not None:
        plt.figure(figsize=figsize)
    if include_equ:
        fermi = Fermi(states3, equ=True, Te=Te, ne=1e19)
        scores, distribution = fermi.calc_mean_distribution(use_power)
        scores_normalized, distribution_normalized = fermi.calc_mean_distribution(use_power)
        plt.plot(scores, distribution, label="equilibrium", marker=".", linewidth=0.8, ms=3)
        plt.plot(scores_normalized, distribution_normalized, label="equilibrium (normalized)", marker=".", linewidth=0.8, ms=3)
    for ne in tqdm(ne_lst):
        fermi = Fermi(states3, equ=False, Te=Te, ne=ne)
        scores, distribution = fermi.calc_mean_distribution(use_power)
        scores_normalized, distribution_normalized = fermi.calc_mean_distribution(use_power)
        plt.plot(scores, distribution, label=fr"$n_e$ = {ne}", marker=".", linewidth=0.8, ms=3)
        plt.plot(scores_normalized, distribution_normalized, label=fr"$n_e$ = {ne} (normalized)", marker=".", linewidth=0.8, ms=3)
    # plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.legend(loc="lower left")
    plt.title(fr"mean distribution for comparing($T_e$ = {Te})")
    plt.yscale("log")
    plt.xlabel(r"$E [\epsilon]$")
    plt.ylabel(r"$P(E)/\rho(E)$  ($\log$ scale)")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def plots_population(
    ne_lst: list[float],
    Te: float = 0.5,
    include_equ: bool = False,
    use_power: bool = True,
    xlim: tuple[float] = None,
    ylim: tuple[float] = None,
    figsize: tuple[float] = None,
    labelsize: int = None,
    titlesize: int = None,
) -> None:
    """
    各neの値におけるフェルミガスモデルを構築し、総エネルギーを横軸にとり、縮退状態を別々で考えた各状態の占有密度を縦軸logスケールでプロットする
    """
    scores = None
    states3 = csv_to_states_from_filename()
    if figsize is not None:
        plt.figure(figsize=figsize)
    if include_equ:
        ne = 1e20
        fermi = Fermi(states3, equ=False, Te=Te, ne=ne)
        scores, population = fermi.calc_population(use_power)
        plt.scatter(scores, population, label=fr"$n_e$={ne} (equilibrium)", s=2, alpha=1.0)
    for ne in tqdm(ne_lst):
        fermi = Fermi(states3, equ=False, Te=Te, ne=ne)
        scores, population = fermi.calc_population(use_power)
        plt.scatter(scores, population, label=fr"$n_e$={ne}", s=2, alpha=1.0)
    # plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=labelsize)
    lgnd = plt.legend(loc="lower left", fontsize=labelsize)
    lgnd.legendHandles[0].set_sizes([9.0])
    lgnd.legendHandles[1].set_sizes([9.0])
    # plt.title(fr"population ($T_e$ = {Te})", fontsize=titlesize)
    plt.title(fr"占有密度分布 ($T_e$ = {Te})", fontsize=titlesize)
    plt.yscale("log")
    # plt.ylabel("population [%] (log scale)")
    plt.xlabel(r"状態 $i$ のエネルギー準位 $E_i$ $[\epsilon]$", fontsize=labelsize)
    plt.ylabel(r"$P(E_i)$  ($\log$ scale)", fontsize=labelsize)
    plt.xlim(xlim)
    plt.ylim(ylim)
    scores_ordered_set = sorted([*set(scores)])
    plt.xticks(scores_ordered_set)
    plt.show()


def plot_population_per_diff(ne: float, Te: float = 0.5, figsize: tuple[float] = None, labelsize: int = None, titlesize: int = None):
    scores = None
    states3 = csv_to_states_from_filename()
    if figsize is not None:
        plt.figure(figsize=figsize)
    fermi = Fermi(states3, equ=False, Te=Te, ne=ne)
    scores_population_tpl = fermi.calc_population_per_diff()
    scores_all = []
    for i, (scores, population) in enumerate(scores_population_tpl):
        scores_all += scores
        plt.scatter(scores, population, label=f"diff {i+1}", s=2, alpha=1.0)
    lgnd = plt.legend(loc="lower left", fontsize=labelsize)
    lgnd.legendHandles[0].set_sizes([9.0])
    lgnd.legendHandles[1].set_sizes([9.0])
    plt.title(fr"diffごとの占有密度分布 ($n_e$ = {ne}, $T_e$ = {Te})", fontsize=titlesize)
    plt.yscale("log")
    # plt.ylabel("population [%] (log scale)")
    plt.xlabel(r"状態 $i$ のエネルギー準位 $E_i$ $[\epsilon]$", fontsize=labelsize)
    plt.ylabel(r"$P(E_i)$  ($\log$ scale)", fontsize=labelsize)
    scores_ordered_set = sorted([*set(scores_all)])
    plt.xticks(scores_ordered_set)
    plt.show()
