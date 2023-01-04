#!/usr/bin/env python
# coding: utf-8
###
# In[233]:


from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import signal


# In[234]:


def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2   #ナイキスト周波数
    wp = fp / fn  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y  

def highpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2   #ナイキスト周波数
    wp = fp / fn  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "high")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y


# In[235]:


allfolder = "./test_fd10/1128/Stop"
#allfolder = "./test_fd10/1025_antyu"
folder = "gankyuzentai_gpu2"
#folder = "gankyuzentai_gpu2"
file=os.listdir(allfolder)
count_file=0
csv_total = 0
totalframe=500
fps=30
wave_f=0.25
cycle=4
stimustart=90
stimufinish = stimustart + fps*(cycle/wave_f)
int(stimufinish)


# In[236]:


for foldername in file:
    
    csvname= allfolder + "/" + foldername +"/"+folder+"/result_tracking.csv"
    csvname_lumi=allfolder + "/" + foldername +"/"+folder+"/result_tracking_luminance.csv"
    csv_input = pd.read_csv(csvname)
    csv_input=csv_input[["num of Kp","x","y","theta"]]
    csv_input_lumi = pd.read_csv(csvname_lumi)
    
    csv_input.plot(title=foldername,subplots=True,y=['num of Kp','x','y','theta'])
    
    #luminaceの順序を降順にする
    if csv_input_lumi['index'].values[0]>csv_input_lumi['index'].values[1]:
        z=csv_input_lumi['index'].values[0]
        csv_input_lumi['index'].values[0]=csv_input_lumi['index'].values[1]
        csv_input_lumi['index'].values[1]=z
    #刺激開始時点の回旋を0にそろえる
    z=csv_input_lumi['index'].values[0]
    stimustart = z
    stimufinish = csv_input_lumi['index'].values[1]
    z=csv_input['theta'].values[z]
    csv_input['theta']=csv_input['theta']-z


    #刺激前の時間をそろえる
    if csv_input_lumi['index'].values[0]>=stimustart:
        dif=csv_input_lumi['index'].values[0]-stimustart
        csv_input=csv_input.drop(range(dif))
    if csv_input_lumi['index'].values[0]<=stimustart:
        dif=stimustart-csv_input_lumi['index'].values[0]
        l=[0]*dif
        data0 = np.array([l,l,l,l]).T
        data0_df=pd.DataFrame(data0)
        data0_df.columns = ["num of Kp","x","y","theta"]
        csv_input = pd.concat([data0_df,csv_input])
    csv_input=csv_input.reset_index()
    #totalframeの長さをそろえる
    if len(csv_input)>=totalframe:
        csv_input.drop(range(totalframe,len(csv_input)))
    if totalframe>=len(csv_input):
        dif=totalframe-len(csv_input)
        l=[0]*dif
        data0 = np.array([l,l,l,l]).T
        data0_df=pd.DataFrame(data0)
        data0_df.columns = ["num of Kp","x","y","theta"]
        csv_input = pd.concat([csv_input,data0_df])


    fp = 1 # 通過域端周波数[Hz]
    fs = 15 # 阻止域端周波数[Hz]
    gpass = 3 # 通過域端最大損失[dB]
    gstop = 40 # 阻止域端最小損失[dB]
    
    # ローパスをする関数を実行
    csv_input['theta'] = lowpass(csv_input['theta'], 30, fp, fs, gpass, gstop)
    csv_input['x'] = lowpass(csv_input['x'], 30, 5, fs, gpass, gstop)
    csv_input['y'] = lowpass(csv_input['y'], 30, 5, fs, gpass, gstop)
    #data_lofilt_dt=lowpass(df['dtheta'], 30, fp, fs, gpass, gstop)

    fp = 0.2 # 通過域端周波数[Hz]
    fs = 0.05 # 阻止域端周波数[Hz]
    gpass = 3 # 通過域端最大損失[dB]
    gstop = 40 # 阻止域端最小損失[dB]

    # ハイパスをする関数を実行
    csv_input['theta'] = highpass(csv_input['theta'], 30, fp, fs, gpass, gstop)
    csv_input['x'] = highpass(csv_input['x'], 30, fp, fs, gpass, gstop)
    csv_input['y'] = highpass(csv_input['y'], 30, fp, fs, gpass, gstop)   

    csv_input=csv_input.reset_index()

    csv_input.to_csv(allfolder + "/" +foldername+'/test7.csv')

    csv = csv_input[["num of Kp","x","y","theta"]]
    csv_total = csv_total+csv
    count_file +=1
    csv_total.to_csv(allfolder + "/" +foldername+'/test_total7.csv')


# In[237]:


print(count_file)
csv_total = csv_total/count_file
csv_total=csv_total.reset_index()     
#csv_total.to_csv(allfolder + '/test.csv')
csv_total.plot(subplots=True,y=['x','y','theta'])
csv_stimulating=csv_total[ stimustart: int(stimufinish+1)]

def fit_func(x, a, b,c):
    return a * np.cos(b*(x/fps*np.pi*2)) - c

fig,ax=plt.subplots(4,1,figsize=(15,10))

#popt, pcov = curve_fit(fit_func, x, y, p0=(0.5, 1.0, 100))
#popt, pcov = curve_fit(fit_func, np.arange(len(csv_total)), csv_total['theta'].values, p0=(1, 3, 0))
popt, pcov = curve_fit(fit_func, np.arange(len(csv_stimulating)), csv_stimulating['theta'].values, p0=(1, 0.25, 0))
#print(f"best-fit parameters = {popt}")
#print(f"covariance = \n{pcov}")

peer = np.sqrt(np.diag(pcov))

d_a = round(peer[0],5)
d_b = round(peer[1],5)
d_c = round(peer[2],5)

print('curve_fittingの結果')
print('振幅:' + str(round(popt[0],4)) + '±' + str(d_a))
print('周波数:' + str(round(popt[1],4)) + '±' + str(d_b))
print('位相:' + str(round(popt[2],4)) + '±' + str(d_c))

#fig, p = plt.subplots(1, 1, sharex=True)
ax[0].plot(np.arange(len(csv_total)),csv_total['x'].values,label='x')
ax[0].set_xlabel('frame')
ax[0].set_ylabel('x px')
ax[0].axvline(x=stimustart, color='red',  linewidth=3)#刺激開始
ax[0].axvline(x=stimufinish, color='red',  linewidth=3)#刺激終了
ax[1].plot(np.arange(len(csv_total)),csv_total['y'].values,label='y')
ax[1].set_xlabel('frame')
ax[1].set_ylabel('y px')
ax[1].axvline(x=stimustart, color='red',  linewidth=3)#刺激開始
ax[1].axvline(x=stimufinish, color='red',  linewidth=3)#刺激終了
ax[2].plot(np.arange(len(csv_total)),csv_total['theta'].values,label='theta')
ax[2].set_xlabel('frame')
ax[2].set_ylabel('amp[degree]')
#ax[2].plot(np.arange(len(csv_total)), fit_func(np.array(np.arange(len(csv_total))), *popt), alpha=0.5, color="crimson")
ax[2].plot(csv_stimulating['index'], fit_func(np.array(np.arange(len(csv_stimulating))), *popt), alpha=0.5, color="crimson")
ax[2].axvline(x=stimustart, color='red',  linewidth=3)#刺激開始
ax[2].axvline(x=stimufinish, color='red',  linewidth=3)#刺激終了
ax[2].set_ylim(-0.05,0.05)

ax[3].plot(np.arange(len(csv_total)),csv_total['num of Kp'].values,label='Kp')
ax[3].set_xlabel('frame')
ax[3].set_ylabel('num of Kp')
#ax[3].axvline(x=30, color='red',  linewidth=3)
#ax[3].axvline(x=(len(a)-stimustart), color='red',  linewidth=3)
ax[3].axhline(y=100, color='red',  linewidth=1)
ax[3].axhline(y=200, color='red',  linewidth=1)
ax[3].axhline(y=300, color='red',  linewidth=1)
#ax[3].set_ylim(0, 500)
"""
for a in ax:
    a.grid()

plt.show()

#p.plot(x, y, "o", markersize=5, markerfacecolor="dodgerblue", markeredgewidth=0.0, fillstyle="full")
#p.plot(x, fit_func(np.array(x), *popt), alpha=0.5, color="crimson")
#p.grid(linewidth=0.5)

#fig.savefig("fit.png", dpi=200, bbox_inches="tight")
"""


# In[ ]:





# In[ ]:




