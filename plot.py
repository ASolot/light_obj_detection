import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
import seaborn as sns
import pandas as pd

import seaborn as sns

import os

from tqdm import tqdm

# LOGS_PATH = "../exp/pwr"
LOG_FILE = "~/output-1559947632.csv"

import pandas as pd 
data = pd.read_csv(LOG_FILE, header=0) 
data['#time (seconds.nanoseconds)'] = np.floor((data['#time (seconds.nanoseconds)'] - data['#time (seconds.nanoseconds)'][0]) * 10) / 10
data.columns = ['time (s)', 'mV', 'mA']
data['power (W)'] = data['mV'] * data['mA'] / 1000.0 / 1000.0

# data = data.set_index('time (s)')

print(data.head())

# ax = sns.lineplot(x="time (s)", y="power (W)", data=data)

t = data.values[:,0]
w = data.values[:,3]

# ## DLA

bg = 5 * 0
en = 5 * 662
t_dla_1024  = t[bg:en]  
w_dla_1024  = w[bg:en]
m = np.mean(w_dla_1024) * (t_dla_1024[-1] - t_dla_1024[0])
print("DLA 1024: ", m , 'J', np.mean(w_dla_1024), 'mean')

bg = 5 * 662
en = 5 * 874
t_dla_512   = t[bg:en]
w_dla_512   = w[bg:en]
m = np.mean(w_dla_512) * (t_dla_512[-1] - t_dla_512[0])
print("DLA 512: ", m , 'J', np.mean(w_dla_512), 'mean')

bg = 5 * 874
en = 5 * 1532
t_dla_orig  = t[bg:en]
w_dla_orig  = w[bg:en]
m = np.mean(w_dla_orig) * (t_dla_orig[-1] - t_dla_orig[0])
print("DLA orig: ", m , 'J', np.mean(w_dla_orig), 'mean')

## EESP
bg = 5 * 1532
en = 5 * 1808
t_esp_1024  = t[bg:en]  
w_esp_1024  = w[bg:en]
m = np.mean(w_esp_1024) * (t_esp_1024[-1] - t_esp_1024[0])
print("EESP 1024: ", m , 'J', np.mean(w_esp_1024), 'mean')

bg = 5 * 1808
en = 5 * 1927
t_esp_512   = t[bg:en]
w_esp_512   = w[bg:en]
m = np.mean(w_esp_512) * (t_esp_512[-1] - t_esp_512[0])
print("EESP 512: ", m , 'J', np.mean(w_esp_512), 'mean')

bg = 5 * 1927
en = 5 * 2190
t_esp_orig  = t[bg:en]
w_esp_orig  = w[bg:en]
m = np.mean(w_esp_orig) * (t_esp_orig[-1] - t_esp_orig[0])
print("EESP Orig: ", m , 'J', np.mean(w_esp_orig), 'mean')

## RESNET
bg = 5 * 2190
en = 5 * 2438
t_res_1024  = t[bg:en]  
w_res_1024  = w[bg:en]
m = np.mean(w_res_1024) * (t_res_1024[-1] - t_res_1024[0])
print("RESNET 1024: ", m , 'J', np.mean(w_res_1024), 'mean')

bg = 5 * 2438
en = 5 * 2522
t_res_512   = t[bg:en]
w_res_512   = w[bg:en]
m = np.mean(w_res_512) * (t_res_512[-1] - t_res_512[0])
print("RESNET 512: ", m , 'J', np.mean(w_res_512), 'mean')

bg = 5 * 2522
en = 5 * 2749
t_res_orig  = t[bg:en]
w_res_orig  = w[bg:en]
m = np.mean(w_res_orig) * (t_res_orig[-1] - t_res_orig[0])
print("RESNET Orig: ", m , 'J', np.mean(w_res_orig), 'mean')

palette = plt.get_cmap('Set1')

# plt.plot(t_dla_1024, w_dla_1024, color=palette(0), label="DLA 1024 - Inf 1024")
# plt.plot(t_dla_512,w_dla_512, color=palette(1), label="DLA 1024 - Inf 512" )
# plt.plot(t_dla_orig, w_dla_orig, color=palette(2), label="DLA 1024 - Inf Orig.")

# plt.plot(t_esp_1024, w_esp_1024, color=palette(3), label="EESP 1024 - Inf 1024")
# plt.plot(t_esp_512,w_esp_512, color=palette(4), label="EESP 1024 - Inf 512" )
# plt.plot(t_esp_orig, w_esp_orig, color=palette(5), label="EESP 1024 - Inf Orig.")

# plt.plot(t_res_1024, w_res_1024, color=palette(6), label="RES 1024 - Inf 1024")
# plt.plot(t_res_512,w_res_512, color=palette(7), label="RES 1024 - Inf 512" )
# plt.plot(t_res_orig, w_res_orig, color=palette(8), label="RES 1024 - Inf Orig.")

# plt.legend(loc='lower left')

# plt.xlabel("Time (s)")
# plt.ylabel("Average Power Consumption (W)")
# plt.show()
# print (w_dla_1024[190*5 : 290*5])


# # Creates two subplots and unpacks the output array immediately
# f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)
# ax1.plot(np.arange(0,100, 0.2), np.array(w_dla_1024[190*5 : 290*5]), color=palette(0), label="DLA 1024 - Inf 1024")
# ax1.set_title("DLA 1024 - Inf 1024")
# ax1.set_ylabel("Pwr Consumption (W)")
# ax2.plot(np.arange(0,100, 0.2), np.array(w_dla_512[38*5 : 138*5]), color=palette(1), label="DLA 1024 - Inf 512" )
# ax2.set_ylabel("Pwr Consumption (W)")
# ax2.set_title("DLA 1024 - Inf 512")
# ax3.plot(np.arange(0,100, 0.2), np.array(w_dla_orig[54*5: 154*5]), color=palette(2), label="DLA 1024 - Inf Orig.")
# ax3.set_xlabel("Time (s)")
# ax3.set_title("DLA 1024 - Inf Orig.")
# ax3.set_ylabel("Pwr Consumption (W))")





