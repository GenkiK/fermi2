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


class RealFermi(object):
    influx_labels: list[str] = ["基底状態からの衝突励起", "基底状態以外からの衝突励起", "衝突脱励起", "放射脱励起"]
    outflux_labels: list[str] = ["衝突励起", "衝突脱励起", "放射脱励起"]
    threshold: float = 1e-9
    loop_assure: float = False

    def __init__(self, states: list[State], Te: float = 1, ne: float = 1e20):
        self.states = states
        self.ne = ne
        self.Te = Te
        self.num_states = len(states)
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
    def power_method(matrix: NDArray[np.float64], eigen: float, threshold: float, loop_assure: bool = False) -> NDArray[np.float64]:
        """
        べき乗法により、最大の固有値に対応する固有ベクトルを求める
        """
        # 初期化
        x = np.zeros(matrix.shape[0])
        x[0] = 1
        cnt = 0
        # 最低10万回はループさせる
        if loop_assure:
            while cnt < 100000:
                if cnt % 10000 == 0:
                    print(f"{cnt}回目")
                y = np.dot(matrix, x)
                cur_eigen = np.dot(y, y) / np.dot(y, x)
                x = y / np.linalg.norm(y)
                cnt += 1
        while cnt < 100000:
            y = np.dot(matrix, x)
            cur_eigen = np.dot(y, y) / np.dot(y, x)
            if np.abs(eigen - cur_eigen) < threshold:
                return x
            x = y / np.linalg.norm(y)
            cnt += 1
        return x

    @staticmethod
    def get_scores(states: list[State]) -> list[int]:
        """重複なしの各エネルギーエネルギー準位を返す"""
        scores_set = set([state.score for state in states])
        return sorted(list(scores_set))

    def make_adj_matrix(self, sym: bool = False) -> None:
        """
        隣接行列を求める
        """
        for i in range(self.num_states):
            for j in range(i + 1, self.num_states):
                if RealFermi.is_connected(self.states[i], self.states[j]):
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
                if RealFermi.is_connected(self.states[i], self.states[j]):
                    # i→jの遷移
                    self.excitation[i, j] = (
                        1.0681088733926924e-14 * self.ne / (self.Te) ** 0.5 * np.exp(-(self.states[j].score - self.states[i].score) / self.Te)
                    )
                    # j→iの遷移
                    self.deexcitation[i, j] = 1.0681088733926924e-14 * self.ne / (self.Te) ** 0.5
                    self.emission[i, j] = 106553.89424678244 * (self.states[j].score - self.states[i].score) ** 3

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
        x = RealFermi.power_method(non_negative_matrix, eigen, self.threshold, RealFermi.loop_assure)
        return x / np.sum(x)

    def calc_distribution(self) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        scoreに対する、縮退状態について和をとった確率密度分布を計算する
        """
        n = self._solve_equation()
        dct = {}
        for rate, state in zip(n, self.states):
            if dct.get(state.score):
                dct[state.score] += rate
            else:
                dct[state.score] = rate
        scores = np.fromiter(dct.keys(), dtype=int)
        distribution = np.fromiter(dct.values(), dtype=float)
        return scores, distribution

    def calc_population(self) -> tuple[list[int], NDArray[np.float64]]:
        """
        「縮退状態を分けて考えた状態」ごとのscoreに対する存在割合を計算する。
        """
        population = self._solve_equation()
        scores_per_state = [state.score for state in self.states]
        return scores_per_state, population

    def calc_mean_distribution(self) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        scoreに対する、縮退状態について和をとり、縮退度で平均した確率密度分布を計算する
        """
        n = self._solve_equation()
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
            plt.stackplot(list(percentage_influx_dict.keys()), *list(percentage_influxes), labels=RealFermi.influx_labels)
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


def real_plots_percentage_fluxes(ne_lst: list[float], Te: float, figsize: tuple[float] = (15, 4), e_num: int = 3) -> None:
    states = csv_to_states_from_filename(f"states{e_num}.csv")
    for ne in ne_lst:
        fig = plt.figure(figsize=figsize)
        fermi = RealFermi(states, Te=Te, ne=ne)
        percentage_influx_dict, percentage_outflux_dict = fermi.calc_percentage_fluxes()
        percentage_influxes = np.array(list(percentage_influx_dict.values())).T
        subfig1 = fig.add_subplot(1, 2, 1)
        subfig1.stackplot(
            list(percentage_influx_dict.keys()),
            *list(percentage_influxes),
            labels=RealFermi.influx_labels,
            colors=["Red", "Orange", "LimeGreen", "DodgerBlue"],
        )
        subfig1.set_xlabel("エネルギー準位 E")
        subfig1.set_ylabel("流入量の割合 [%]")
        subfig1.set_title(fr"エネルギー準位ごとの流入量 ($n_e$={ne}" + r"$m^{-3}$" + f", $T_e$={Te}$eV$, 電子数={e_num})")
        subfig1.set_xticks(list(percentage_influx_dict.keys()))
        subfig1.set_xmargin(0)
        subfig1.set_ymargin(0)
        subfig1.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=0.5)

        percentage_outfluxes = np.array(list(percentage_outflux_dict.values())).T
        subfig2 = fig.add_subplot(1, 2, 2)
        subfig2.stackplot(
            list(percentage_outflux_dict.keys()),
            *list(percentage_outfluxes),
            labels=RealFermi.outflux_labels,
            colors=["Red", "LimeGreen", "DodgerBlue"],
        )
        subfig2.set_xlabel("エネルギー準位 E")
        subfig2.set_ylabel("流出量の割合 [%]")
        subfig2.set_title(fr"エネルギー準位ごとの流出量 ($n_e$={ne}" + r"$m^{-3}$" + f", $T_e$={Te}$eV$, 電子数={e_num})")
        subfig2.set_xticks(list(percentage_outflux_dict.keys()))
        subfig2.set_xmargin(0)
        subfig2.set_ymargin(0)
        subfig2.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=0.5)
        plt.show()


def real_plots_dist(
    ne_lst: list[float],
    Te: float,
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
    for ne in tqdm(ne_lst):
        fermi = RealFermi(states3, Te=Te, ne=ne)
        scores, population = fermi.calc_distribution()
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


def real_plots_mean_dist(
    ne_lst: list[float],
    Te: float,
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
    for ne in tqdm(ne_lst):
        fermi = RealFermi(states3, Te=Te, ne=ne)
        scores, distribution = fermi.calc_mean_distribution()
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


def real_plots_mean_dist_compare(
    ne_lst: list[float],
    Te: float,
    xlim: tuple[float] = None,
    ylim: tuple[float] = None,
    figsize: tuple[float] = None,
) -> None:
    states3 = csv_to_states_from_filename()
    if figsize is not None:
        plt.figure(figsize=figsize)
    for ne in tqdm(ne_lst):
        fermi = RealFermi(states3, Te=Te, ne=ne)
        scores, distribution = fermi.calc_mean_distribution()
        scores_normalized, distribution_normalized = fermi.calc_mean_distribution()
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


def real_plots_population(
    ne_lst: list[float],
    Te: float,
    xlim: tuple[float] = None,
    ylim: tuple[float] = None,
    figsize: tuple[float] = None,
    labelsize: int = None,
    titlesize: int = None,
) -> None:
    """
    各neの値におけるフェルミガスモデルを構築し、総エネルギーを横軸にとり、縮退状態を別々で考えた各状態の占有密度を縦軸logスケールでプロットする
    """
    scores_per_state = None
    states3 = csv_to_states_from_filename()
    if figsize is not None:
        plt.figure(figsize=figsize)
    for ne in tqdm(ne_lst):
        fermi = RealFermi(states3, Te=Te, ne=ne)
        scores_per_state, population = fermi.calc_population(True)
        plt.scatter(scores_per_state, population, label=fr"$n_e$={ne}", s=2, alpha=1.0)
    # plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=labelsize)
    plt.legend(loc="lower left", fontsize=labelsize)
    plt.title(fr"占有密度分布 ($T_e$ = {Te})", fontsize=titlesize)
    plt.yscale("log")
    plt.xlabel(r"状態 $i$ のエネルギー準位 $E_i$ $[\epsilon]$", fontsize=labelsize)
    plt.ylabel(r"$P(E_i)$  ($\log$ scale)", fontsize=labelsize)
    plt.xlim(xlim)
    plt.ylim(ylim)
    scores_ordered_set = sorted([*set(scores_per_state)])
    plt.xticks(scores_ordered_set)
    plt.show()


def real_plot_population_per_diff(ne: float, Te: float, figsize: tuple[float] = None, labelsize: int = None, titlesize: int = None, e_num: int = 3):
    scores = None
    states = csv_to_states_from_filename(f"states{e_num}.csv")
    if figsize is not None:
        plt.figure(figsize=figsize)
    fermi = RealFermi(states, Te=Te, ne=ne)
    scores_population_tpl = fermi.calc_population_per_diff()
    scores_all = []
    for i, (scores, population) in enumerate(scores_population_tpl):
        scores_all += scores
        plt.scatter(scores, population, label=f"diff {i+1}", s=2, alpha=1.0)
    lgnd = plt.legend(loc="lower left", fontsize=labelsize)
    lgnd.legendHandles[0].set_sizes([9.0])
    lgnd.legendHandles[1].set_sizes([9.0])
    plt.title(fr"diffごとの占有密度分布 ($n_e$ = {ne}, $T_e$ = {Te}, 電子数 = {e_num})", fontsize=titlesize)
    plt.yscale("log")
    # plt.ylabel("population [%] (log scale)")
    plt.xlabel(r"状態 $i$ のエネルギー準位 $E_i$ $[\epsilon]$", fontsize=labelsize)
    plt.ylabel(r"$P(E_i)$  ($\log$ scale)", fontsize=labelsize)
    scores_ordered_set = sorted([*set(scores_all)])
    plt.xticks(scores_ordered_set)
    plt.show()


def real_plot_population_and_mean_distribution(ne: float, Te: float, e_num: int = 3, figsize: tuple[int] = None, ylim: tuple = None):
    states = csv_to_states_from_filename(f"states{e_num}.csv")
    if figsize is not None:
        plt.figure(figsize=figsize)
    fermi = RealFermi(states, Te=Te, ne=ne)
    scores_per_state, population = fermi.calc_population()
    plt.scatter(scores_per_state, population, label="占有密度", s=2, alpha=1.0)

    present_score = 0
    dct = {}
    degeneracy = 0
    for rate, score in zip(population, scores_per_state):
        if score != present_score:
            if degeneracy != 0:
                dct[score - 1] /= degeneracy

            dct[score] = rate
            present_score = score
            degeneracy = 1
        else:
            dct[score] += rate
            degeneracy += 1
    dct[scores_per_state[-1]] /= degeneracy
    scores = np.fromiter(dct.keys(), dtype=int)
    mean_dist = np.fromiter(dct.values(), dtype=float)
    plt.plot(scores, mean_dist, label="縮退度で平均した占有密度", marker=".", linewidth=1, ms=4, color="black")
    plt.legend(loc="lower left")
    # plt.title(fr"占有密度分布 ($T_e$={Te}[$eV$], $n_e$={ne}[$m^{-3}$], 電子数={e_num})")
    plt.title(fr"占有密度分布 ($n_e$={ne}" + r"$m^{-3}$" + f", $T_e$={Te}$eV$, 電子数={e_num})")
    plt.yscale("log")
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel(r"状態 $i$ のエネルギー準位 $E_i$ $[\epsilon]$", fontsize=15)
    plt.ylabel(r"$P(E_i)$  ($\log$ scale)", fontsize=15)
    scores_ordered_set = sorted([*set(scores)])
    plt.xticks(scores_ordered_set)
    plt.show()
