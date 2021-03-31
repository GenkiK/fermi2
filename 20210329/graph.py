# %%
# 初期設定
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import subprocess
import os

# %%
# ファイルが存在しないorデータが足りない場合,c++実行
n = 8  # 電子の数[個]
lim_size = 5  # 最大の総エネルギー[ε]の個数 [(n-1)*n / 2,  (n-1)*n / 2 + lim_size]

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
