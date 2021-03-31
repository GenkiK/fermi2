# %%
# 初期設定
import xarray as xr
import matplotlib.pyplot as plt
import pyspectra
import numpy as np
import csv
import pandas as pd
import subprocess

from scipy.stats import poisson

# %%
# c++実行
n = 3
lim_size = 100
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
