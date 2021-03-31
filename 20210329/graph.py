# %%
# 初期設定
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import subprocess

# %%
# c++実行
n = 3  # 電子の数[個]
lim_size = 100  # 最大の総エネルギー[ε]の個数 [(n-1)!    (n-1)! + lim_size]
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
