# %%
# 初期設定
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import csv
import pandas as pd
import subprocess
import os

# %%
# ファイルが存在しないorデータが足りない場合,c++実行
n = 8  # 電子の数[個]
lim_size = 50  # 最大の総エネルギー[ε]の個数 [(n-1)*n / 2,  (n-1)*n / 2 + lim_size]

filename = "./state_count" + str(n) + ".csv"
if os.path.exists(filename):
    line_count = (
        int(subprocess.check_output(["wc", "-l", filename]).decode().split(" ")[0]) - 1
    )
    if line_count < lim_size:
        cmd = "./a.out " + str(n) + " " + str(lim_size)
        subprocess.run(cmd.split())

# %%
filename = "./state_count" + str(n) + ".csv"
csv_input = pd.read_csv(filepath_or_buffer=filename, sep=",")

# %%
plt.plot(csv_input.level, csv_input.nums)
plt.yscale("log")
plt.title("n = " + str(n))
plt.xlabel("ε")
plt.ylabel("個/ε")
# %%
pair_filename = "./pair_count" + str(n) + ".csv"
pairs = pd.read_csv(filepath_or_buffer=pair_filename, sep=",")
pairs

# %%
memo = np.zeros((len(pairs), len(pairs)))
min_level = csv_input.level[0]
for i in range(len(csv_input)):  # 下準位
    under = csv_input.level[i]
    for j in range(i + 1, len(csv_input)):  # 上準位
        upper = csv_input.level[j]
        if pairs[str(upper)][under] == 0:
            continue
        memo[under][upper] = csv_input.nums[j] * csv_input.nums[i]  # jからiへの遷移数
        memo[under][upper] = pairs[str(upper)][under] / memo[under][upper]
        memo[upper][under] = memo[under][upper]
    memo[under][under] = (
        memo[under + 1][under] if i + 1 < len(csv_input) else memo[under - 1][under]
    )
memo = np.where(memo == 0, 1e-2, memo)
x = np.arange(len(pairs))
y = np.arange(len(pairs))
# %%
plt.pcolormesh(x, y, memo, norm=colors.LogNorm(vmin=memo.min(), vmax=memo.max()))
plt.xlim(csv_input.level[0], csv_input.level[len(csv_input) - 1])
plt.ylim(csv_input.level[0], csv_input.level[len(csv_input) - 1])
plt.colorbar()
plt.title("n = " + str(n))
#

# %%
csv_input.level[min_level]
# %%
