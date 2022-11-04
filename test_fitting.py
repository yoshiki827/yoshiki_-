
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
"""
x = [
    0, 10,  20,  30,  40,  50,  60,  70,  80,  90, 100, 110, 120, 130, 140, 150, 160, 170,
    180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350
]
y = [
    0.82588192, 0.85386846, 0.87985536, 0.90294279, 0.92744663, 0.95140891,
    0.96978054, 0.98171169, 0.99232641, 1.,       0.99891687, 0.99444269,
    0.9832864,  0.96853077, 0.95128393, 0.93248738, 0.90825015, 0.88215494,
    0.85495159, 0.82853977, 0.80223626, 0.78005699, 0.75953575, 0.73908117,
    0.72210095, 0.70438753, 0.69248971, 0.68112513, 0.68318308, 0.69080669,
    0.7034627,  0.71821833, 0.73359051, 0.74643815, 0.76822583, 0.7936295
]
"""

allfolder = "./test_fd10/1025_antyu"
folder = "gankyuzentai_gpu2"
file=os.listdir(allfolder)
count_file=0
csv_total = 0
for foldername in file:
    csvname= allfolder + "/" + foldername +"/"+folder+"/result_tracking.csv"
    csv_input = pd.read_csv(csvname)
    csv = csv_input[["num of Kp","x","y","theta"]]
    csv_total = csv_total+csv
    print(csvname)
    print(csv_total)
    count_file +=1

print(count_file)
csv_total = csv_total/count_file     
csv_total.plot(subplots=True,y=['x','y','theta'])

def fit_func(x, a, b, c):
    return a * np.cos(b*(x/360.0*np.pi*2 - c))

fig,ax=plt.subplots(4,1,figsize=(15,10))

#popt, pcov = curve_fit(fit_func, x, y, p0=(0.5, 1.0, 100))
popt, pcov = curve_fit(fit_func, np.arange(len(csv_total)), csv_total['theta'].values, p0=(1, 3, 0))
print(f"best-fit parameters = {popt}")
print(f"covariance = \n{pcov}")

#fig, p = plt.subplots(1, 1, sharex=True)
ax[0].plot(np.arange(len(csv_total)),csv_total['x'].values,label='x')
ax[0].set_xlabel('frame')
ax[0].set_ylabel('x ')
ax[1].plot(np.arange(len(csv_total)),csv_total['y'].values,label='y')
ax[1].set_xlabel('frame')
ax[1].set_ylabel('y px')
ax[2].plot(np.arange(len(csv_total)),csv_total['theta'].values,label='theta')
ax[2].set_xlabel('frame')
ax[2].set_ylabel('amp[degree]')
ax[2].plot(np.arange(len(csv_total)), fit_func(np.array(np.arange(len(csv_total))), *popt), alpha=0.5, color="crimson")
ax[3].plot(np.arange(len(csv_total)),csv_total['num of Kp'].values,label='Kp')
ax[3].set_xlabel('frame')
ax[3].set_ylabel('num of Kp')
#ax[3].axvline(x=30, color='red',  linewidth=3)
#ax[3].axvline(x=(len(a)-90), color='red',  linewidth=3)
ax[3].axhline(y=100, color='red',  linewidth=1)
ax[3].axhline(y=200, color='red',  linewidth=1)
ax[3].axhline(y=300, color='red',  linewidth=1)
#ax[3].set_ylim(0, 500)
for a in ax:
    a.grid()

plt.show()

#p.plot(x, y, "o", markersize=5, markerfacecolor="dodgerblue", markeredgewidth=0.0, fillstyle="full")
#p.plot(x, fit_func(np.array(x), *popt), alpha=0.5, color="crimson")
#p.grid(linewidth=0.5)

#fig.savefig("fit.png", dpi=200, bbox_inches="tight")