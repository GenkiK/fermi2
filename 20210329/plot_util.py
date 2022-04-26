import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("Paired")
colors = sns.color_palette("Paired")
skyblue = colors[0]
blue = colors[1]
lightgreen = colors[2]
green = colors[3]
pink = colors[4]
red = colors[5]
lightorange = colors[6]
orange = colors[7]

# 数値解からフィッティングにより得たパラメータ
rho_0_ = 11.282243614633337
eps_0_ = 4.993125454624417
# FAC
rho_0 = 0.3583076359836078
eps_0 = 150.14510859857336

eV2eps = eps_0_ / eps_0
eps2eV = 1 / eV2eps

a_0 = 0.529 * 1e-10  # ボーア半径[m]
E_H = 27.2  # ハートリーエネルギー[eV]
c = 299792458  # 光速[m/s]
alpha = 1 / 137  # 微細構造定数[無次元]

beta = 2 ** 2.5 / 3 * np.pi ** 0.5 * alpha * a_0 ** 2 * c * E_H ** 0.5
gamma = 4 / 3 * alpha ** 4 * c / a_0 / E_H ** 3

scaled_S_0 = 0.014707183753929784
p = 0.04358816037996319
scaled_S_all = 0.025929592291995655
N = 12


def calc_approx_popu(E_arr, ne, Te, all=False):
    T_eff = (Te * eps_0) / (Te + eps_0)
    if all:
        return 1 / (6 * eps_0 ** 4) * beta / gamma * ne / Te ** 0.5 * (1 + (Te / eps_0) ** 4) * np.exp(-E_arr / T_eff)
    return 1 / (6 * eps_0 ** 4) * beta / gamma * ne / Te ** 0.5 * (1 + (T_eff / eps_0) ** 4) * np.exp(-E_arr / T_eff)


def calc_mean_dist(scores_per_state, popu):
    pre_score = 0
    dct = {}
    degeneracy = 0
    for rate, score in zip(popu, scores_per_state):
        if score != pre_score:
            # １つ前のscoreの割合に対して、縮退度で平均をとる
            if degeneracy != 0:
                dct[score - 1] /= degeneracy

            dct[score] = rate
            pre_score = score
            degeneracy = 1
        else:
            dct[score] += rate
            degeneracy += 1
    dct[scores_per_state[-1]] /= degeneracy
    mean_distribution = np.fromiter(dct.values(), dtype=float)
    return mean_distribution


def plot_popu_anal_mean(scores, scores_per_state, population, Te, ne, all=False, with_legend=False):
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.set_title(
        "$T_\mathrm{e} =\; $" + f"${Te}$" + "$\:\:\mathrm{eV},\;\; n_\mathrm{e} = $" + f"$\mathrm{{{ne}}}$" + "$\:\:\mathrm{m^{-3}}$", fontsize=21
    )
    approx_popu = calc_approx_popu(scores * eps2eV, ne=ne, Te=Te, all=all)
    approx_popu[0] = 1 - np.sum(approx_popu[1:])
    ax.plot(scores * eps2eV, approx_popu, color=orange, label="$n(E) \;\;\mathrm{(analytical)}$", alpha=0.9, linewidth=2.5)
    ax.plot(
        scores * eps2eV,
        calc_mean_dist(scores_per_state, population),
        "-",
        color="black",
        label="$\overline{n_i} \;\;\mathrm{(numerical)}$",
        alpha=0.4,
        linewidth=2.5,
    )
    ax.plot(
        (scores_per_state - scores_per_state[0]) * eps2eV, population, ".", color="black", ms=3.5, alpha=0.8, label="$n_i \;\;\mathrm{(numerical)}$"
    )
    ax.set_xlabel(r"$E \:\: \mathrm{[eV]}$", fontsize=22)
    ax.set_ylabel(r"$n$", fontsize=22)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(0, 900, 200))
    ax.set_yticks([1e-50, 1e-37, 1e-24, 1e-11, 1e2])
    ax.set_ylim(bottom=1e-50, top=1e2)
    if with_legend:
        leg = ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=22, markerscale=2.5)
        leg.get_frame().set_linewidth(2)
    fig.show()
    if with_legend:
        if all:
            fig.savefig(f"figs/{str(ne).replace('+', '')}_{Te}_popu_and_anal_and_mean_all_with_legend.pdf", bbox_inches="tight")
        else:
            fig.savefig(f"figs/{str(ne).replace('+', '')}_{Te}_popu_and_anal_and_mean_with_legend.pdf", bbox_inches="tight")
    else:
        if all:
            fig.savefig(f"figs/{str(ne).replace('+', '')}_{Te}_popu_and_anal_and_mean_all.pdf", bbox_inches="tight")
        else:
            fig.savefig(f"figs/{str(ne).replace('+', '')}_{Te}_popu_and_anal_and_mean.pdf", bbox_inches="tight")


def plot_flows(influx_dict, outflux_dict, Te, ne, all=False, with_legend=False):
    fig = plt.figure(figsize=(6, 5.5))
    plt.subplots_adjust(hspace=0)
    percentage_influxes = np.array(list(influx_dict.values())).T
    scores = np.array(list(influx_dict.keys()))
    scores -= np.min(scores)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.stackplot(
        scores / eV2eps,
        np.array(percentage_influxes),
        labels=[
            "$C n_\mathrm{e} \mathrm{\;from\; the\; ground\; level}$",
            "$C n_\mathrm{e}\mathrm{\;from\; excited\; levels}$",
            "$F n_\mathrm{e}$",
            "$A$",
        ],
        colors=[pink, lightorange, lightgreen, skyblue],
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("inflow", fontsize=21)
    ax1.set_title(
        "$T_\mathrm{e} =\; $" + f"${Te}$" + "$\:\:\mathrm{eV},\;\; n_\mathrm{e} = $" + f"$\mathrm{{{ne}}}$" + "$\:\:\mathrm{m^{-3}}$",
        fontsize=22,
        pad=8,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(0, scores[-1] / eV2eps)
    ax1.set_ylim(0, 1)
    ax1.set_xmargin(0)
    ax1.set_ymargin(0)
    ax1.spines["bottom"].set_linewidth(2.5)
    ax1.spines["top"].set_linewidth(2.5)
    if with_legend:
        handles1, labels1 = ax1.get_legend_handles_labels()
        legend1 = ax1.legend(handles1[::-1], labels1[::-1], fontsize=21, bbox_to_anchor=(1.03, 1), loc="upper left", borderaxespad=0)
        legend1.get_frame().set_linewidth(2)

    percentage_outfluxes = np.array(list(outflux_dict.values())).T
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.stackplot(
        scores / eV2eps,
        np.array(percentage_outfluxes),
        labels=["$C n_\mathrm{e}$", "$F n_\mathrm{e}$", "$A$"],
        colors=[orange, lightgreen, skyblue],
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_xlabel("$E\;\mathrm{[eV]}$", fontsize=22)
    ax2.set_ylabel("outflow", fontsize=21)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim(0, scores[-1] / eV2eps)
    ax2.set_ylim(0, 1)
    ax2.set_xmargin(0)
    ax2.set_ymargin(0)
    ax2.spines["top"].set_linewidth(2.5)
    ax2.spines["bottom"].set_linewidth(2.5)
    if with_legend:
        handles2, labels2 = ax2.get_legend_handles_labels()
        legend2 = ax2.legend(handles2[::-1], labels2[::-1], fontsize=21, bbox_to_anchor=(1.03, 1), loc="upper left", borderaxespad=0)
        legend2.get_frame().set_linewidth(2)
    fig.show()
    if with_legend:
        if all:
            fig.savefig(f'figs/{str(ne).replace("+", "")}_{Te}_flow_all_with_legend.pdf', bbox_inches="tight")
        else:
            fig.savefig(f'figs/{str(ne).replace("+", "")}_{Te}_flow_with_legend.pdf', bbox_inches="tight")
    else:
        if all:
            fig.savefig(f'figs/{str(ne).replace("+", "")}_{Te}_flow_all.pdf', bbox_inches="tight")
        else:
            fig.savefig(f'figs/{str(ne).replace("+", "")}_{Te}_flow.pdf', bbox_inches="tight")


from scipy.special import gammainc


def calc_approx_intensity_density(I: float, ne: float, Te: float, E_range: float):
    T_eff = Te * eps_0 / (eps_0 + Te)
    T_ = T_eff / eps_0
    B = scaled_S_0 * beta / (6 * eps_0 ** 4) * ne / Te ** 0.5 * (1 + (T_eff / eps_0) ** 4)
    gamma = gammainc(3 * T_ + 1, E_range / eps_0)
    return 1 / 2 * N * rho_0 * I ** (-T_ - 1) * B ** T_ * T_eff * eps_0 ** (3 * T_ + 1) * gamma


def calc_approx_intensity_density_all(I: float, ne: float, Te: float, E_range: float):
    T_eff = Te * eps_0 / (eps_0 + Te)
    T_ = T_eff / eps_0
    B_ = scaled_S_all * beta / (6 * eps_0 ** 4) * ne / Te ** 0.5 * (1 + (Te / eps_0) ** 4)
    gamma = gammainc(6 * T_ + 1, E_range / eps_0)
    return 1 / 2 * p * rho_0 * I ** (-2 * T_ - 1) * B_ ** (2 * T_) * T_eff * eps_0 ** (6 * T_ + 1) * gamma


def plot_rho_Is(A_lst, population_lst, ne_lst, Te, bin_num=30, all=False, is_wide=False, with_legend=False):
    my_colors = [blue, red, green]
    E_range = eps_0
    color_idx = 0

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.set_title(f"$T_\mathrm{{e}}=\mathrm{{{int(Te)}}} \:\:\mathrm{{eV}}$", fontsize=22, pad=6)
    for A, population, ne in zip(A_lst, population_lst, ne_lst):
        I_mat = A * population
        I_arr = I_mat[I_mat > 0]
        I_arr_log = np.log(I_arr)
        bins = np.logspace(start=np.min(I_arr_log), stop=np.max(I_arr_log), num=bin_num, base=np.e)
        hist = np.histogram(I_arr, bins=bins)[0] / np.diff(bins)
        if all:
            approx_intensity_densities = calc_approx_intensity_density_all(I=bins[:-1], ne=ne, Te=Te, E_range=E_range)
        else:
            approx_intensity_densities = calc_approx_intensity_density(I=bins[:-1], ne=ne, Te=Te, E_range=E_range)
        ax.bar(
            bins[:-1],
            hist,
            width=np.diff(bins),
            align="edge",
            color=my_colors[color_idx],
            alpha=0.25,
            label=f"$n_\mathrm{{e}}=\mathrm{{{ne}}}\:\:\mathrm{{m^{{-3}}}}\:\:\mathrm{{(numerical)}}$",
        )
        ax.plot(
            bins[:-1],
            approx_intensity_densities,
            "-",
            linewidth=2,
            color=my_colors[color_idx],
            label=f"$n_\mathrm{{e}}=\mathrm{{{ne}}}\:\:\mathrm{{m^{{-3}}}}\:\:\mathrm{{(analytical)}}$",
        )
        color_idx += 1
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$I\:\:\mathrm{[s^{-1}]}$", fontsize=22)
    ax.set_ylabel(r"$\rho_I(I)\:\:\mathrm{[s]}$", fontsize=22)
    if is_wide:
        ax.set_xlim(1e-18, 1e10)
        ax.set_xticks([1e-12, 1e-1, 1e-4, 1e3, 1e10])
        ax.set_xticks([1e-18, 1e-11, 1e-4, 1e3, 1e10])
        ax.set_ylim(bottom=1e-9, top=1e21)
        ax.set_yticks([1e-9, 1e-3, 1e3, 1e9, 1e15, 1e21])
        if with_legend:
            fig.legend(bbox_to_anchor=(0.96, 0.99), loc="upper left", borderaxespad=0, fontsize=21)
            if all:
                fig.savefig(f"figs/{int(Te)}_all_rho_Is_with_legend_wide_numerical.pdf", bbox_inches="tight")
            else:
                fig.savefig(f"figs/{int(Te)}_rho_Is_with_legend_wide_numerical.pdf", bbox_inches="tight")
        else:
            if all:
                fig.savefig(f"figs/{int(Te)}_all_rho_Is_wide_numerical.pdf", bbox_inches="tight")
            else:
                fig.savefig(f"figs/{int(Te)}_rho_Is_wide_numerical.pdf", bbox_inches="tight")
    else:
        ax.set_xlim(1e-12, 1e10)
        ax.set_ylim(bottom=1e-9, top=1e15)
        ax.set_yticks([1e-9, 1e-3, 1e3, 1e9, 1e15])
        if with_legend:
            fig.legend(bbox_to_anchor=(0.96, 0.99), loc="upper left", borderaxespad=0, fontsize=21)
            if all:
                fig.savefig(f"figs/{int(Te)}_all_rho_Is_with_legend_numerical.pdf", bbox_inches="tight")
            else:
                fig.savefig(f"figs/{int(Te)}_rho_Is_with_legend_numerical.pdf", bbox_inches="tight")
        else:
            if all:
                fig.savefig(f"figs/{int(Te)}_all_rho_Is_numerical.pdf", bbox_inches="tight")
            else:
                fig.savefig(f"figs/{int(Te)}_rho_Is_numerical.pdf", bbox_inches="tight")
    fig.show()


def plot_rho_I(A, population, ne, Te, bin_num=30, all=False, is_wide=False, with_legend=False):
    my_colors = [blue, red, green]
    E_range = eps_0

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.set_title(
        "$T_\mathrm{e} =\; $" + f"${Te}$" + "$\:\:\mathrm{eV},\;\; n_\mathrm{e} = $" + f"$\mathrm{{{ne}}}$" + "$\:\:\mathrm{m^{-3}}$",
        fontsize=22,
        pad=6,
    )
    I_mat = A * population
    I_arr = I_mat[I_mat > 0]
    I_arr_log = np.log(I_arr)
    bins = np.logspace(start=np.min(I_arr_log), stop=np.max(I_arr_log), num=bin_num, base=np.e)
    hist = np.histogram(I_arr, bins=bins)[0] / np.diff(bins)
    if all:
        approx_intensity_densities = calc_approx_intensity_density_all(I=bins[:-1], ne=ne, Te=Te, E_range=E_range)
    else:
        approx_intensity_densities = calc_approx_intensity_density(I=bins[:-1], ne=ne, Te=Te, E_range=E_range)
    ax.bar(
        bins[:-1],
        hist,
        width=np.diff(bins),
        align="edge",
        color=my_colors[0],
        alpha=0.25,
        label="$\mathrm{numerical}$",
    )
    ax.plot(
        bins[:-1],
        approx_intensity_densities,
        "-",
        linewidth=2,
        color=my_colors[1],
        label="$\mathrm{analytical}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$I\:\:\mathrm{[s^{-1}]}$", fontsize=22)
    ax.set_ylabel(r"$\rho_I(I)\:\:\mathrm{[s]}$", fontsize=22)
    if is_wide:
        ax.set_xlim(1e-18, 1e10)
        ax.set_xticks([1e-12, 1e-1, 1e-4, 1e3, 1e10])
        ax.set_xticks([1e-18, 1e-11, 1e-4, 1e3, 1e10])
        ax.set_ylim(bottom=1e-9, top=1e21)
        ax.set_yticks([1e-9, 1e-3, 1e3, 1e9, 1e15, 1e21])
        if with_legend:
            fig.legend(bbox_to_anchor=(0.96, 0.99), loc="upper left", borderaxespad=0, fontsize=21)
            if all:
                fig.savefig(f"figs/{int(Te)}_all_rho_I_with_legend_wide_numerical.pdf", bbox_inches="tight")
            else:
                fig.savefig(f"figs/{int(Te)}_rho_I_with_legend_wide_numerical.pdf", bbox_inches="tight")
        else:
            if all:
                fig.savefig(f"figs/{int(Te)}_all_rho_I_wide_numerical.pdf", bbox_inches="tight")
            else:
                fig.savefig(f"figs/{int(Te)}_rho_I_wide_numerical.pdf", bbox_inches="tight")
    else:
        ax.set_xlim(1e-12, 1e10)
        ax.set_ylim(bottom=1e-9, top=1e15)
        ax.set_yticks([1e-9, 1e-3, 1e3, 1e9, 1e15])
        if with_legend:
            fig.legend(bbox_to_anchor=(0.96, 0.99), loc="upper left", borderaxespad=0, fontsize=21)
            if all:
                fig.savefig(f"figs/{int(Te)}_all_rho_I_with_legend_numerical.pdf", bbox_inches="tight")
            else:
                fig.savefig(f"figs/{int(Te)}_rho_I_with_legend_numerical.pdf", bbox_inches="tight")
        else:
            if all:
                fig.savefig(f"figs/{int(Te)}_all_rho_I_numerical.pdf", bbox_inches="tight")
            else:
                fig.savefig(f"figs/{int(Te)}_rho_I_numerical.pdf", bbox_inches="tight")
    fig.show()
