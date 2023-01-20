#!/usr/bin/env python
# coding: utf-8
###
# In[233]:

from numpy.fft import fft,ifft
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import statistics
import math
from scipy import signal
from pandas import DataFrame

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

def fit_func(x, a, b,c):
    return a * np.sin(b*(x/fps*np.pi*2 - c)) 
# In[235]:


allfolder = "./test_fd1/0109waranabe_zaimaemuki_2pole_01"
#allfolder = "./test_fd10/1025_antyu"
folder = "kaiseki0113"
#folder = "gankyuzentai_gpu2"
file=os.listdir(allfolder)
count_file=0
csv_total = 0
totalframe=450
fps=30
wave_f=0.25
cycle=3
stimustart=60
stimufinish = stimustart + fps*(cycle/wave_f)
#stimufinish = 120
int(stimufinish)
list0=[]
list=[]
list2=[]
list3=[]


# In[236]:


for foldername in file:
    base, ext = os.path.splitext(foldername)
    if ext != ".mp4":
        continue

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








    #直線ドリフト除去
    #csv_input['num of Kp'] = signal.detrend(csv_input['num of Kp'])
    csv_input['x'] = signal.detrend(csv_input['x'])
    csv_input['y'] = signal.detrend(csv_input['y'])
    csv_input['theta'] = signal.detrend(csv_input['theta'])
    
    #csv_input.plot(title="detrend",subplots=True,y=['num of Kp','x','y','theta'])

    sr=fps
    ts=1.0/sr
    fc=1
    fc_upper=0.2


    X = fft(csv_input['x'].values)
    N_x=len(X)
    n = np.arange(N_x)
    T=N_x/sr
    freq=n/T
    
    # 正規化 + 交流成分2倍
    X = X/(N_x/2)
    X[0] = X[0]/2
    
    X2 = X.copy()


    Y = fft(csv_input['y'].values)
    N_y=len(Y)
    n = np.arange(N_y)
    T=N_y/sr
    freq=n/T
    
    # 正規化 + 交流成分2倍
    Y = Y/(N_y/2)
    Y[0] = Y[0]/2
    
    Y2 = Y.copy()

    theta = fft(csv_input['theta'].values)
    N_t=len(theta)
    n = np.arange(N_t)
    T=N_t/sr
    freq=n/T
    
    # 正規化 + 交流成分2倍
    theta = theta/(N_t/2)
    theta[0] = theta[0]/2
    
    T2 = theta.copy()


    #Kp2[(freq > fc)] = 0
    X2[(freq > fc)] = 0
    Y2[(freq > fc)] = 0
    T2[(freq > fc)] = 0
    
    #Kp2[(freq < fc_upper)&(freq > 1/(4*totalframe/fps))] = 0
    X2[(freq < fc_upper)&(freq > 1/(4*totalframe/fps))] = 0
    Y2[(freq < fc_upper)&(freq > 1/(4*totalframe/fps))] = 0
    T2[(freq < fc_upper)&(freq > 1/(4*totalframe/fps))] = 0

    #kp2 = ifft(Kp2)
    x2 = ifft(X2)
    y2 = ifft(Y2)
    t2 = ifft(T2)
    
    #csv_input['num of Kp'] = np.real(kp2*N_k)
    csv_input['x'] = np.real(x2*N_x)
    csv_input['y'] = np.real(y2*N_y)
    csv_input['theta'] = np.real(t2*N_t)
    
    csv_input.plot(title="lowpass",subplots=True,y=['num of Kp','x','y','theta'])

    """
    fp = 2 # 通過域端周波数[Hz]
    fs = 8 # 阻止域端周波数[Hz]
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

    #csv_input=csv_input.reset_index()
    """

    #刺激前の時間をそろえる
    if csv_input_lumi['index'].values[0]>=stimustart:
        dif=csv_input_lumi['index'].values[0]-stimustart
        csv_input=csv_input.drop(range(dif))
    if csv_input_lumi['index'].values[0]<=stimustart:
        dif=stimustart-csv_input_lumi['index'].values[0]
        l1= [csv_input['num of Kp'].values[0]] * dif
        l2= [csv_input['x'].values[0] ]* dif
        l3= [csv_input['y'].values[0]] * dif
        l4= [csv_input['theta'].values[0]] * dif

        data0 = np.array([l1,l2,l3,l4]).T
        data0_df=pd.DataFrame(data0)
        data0_df.columns = ["num of Kp","x","y","theta"]
        csv_input = pd.concat([data0_df,csv_input])
    csv_input=csv_input.reset_index()
    #totalframeの長さをそろえる
    if len(csv_input)>=totalframe:
        csv_input = csv_input.drop(range(totalframe,len(csv_input)))

    if totalframe>=len(csv_input):
        dif=totalframe-len(csv_input)
        Len=len(csv_input)-1
        l1= [csv_input['num of Kp'].values[Len]] * dif
        l2= [csv_input['x'].values[Len] ]* dif
        l3= [csv_input['y'].values[Len]] * dif
        l4= [csv_input['theta'].values[Len]] * dif
        
        data0 = np.array([l1,l2,l3,l4]).T
        data0_df=pd.DataFrame(data0)
        data0_df.columns = ["num of Kp","x","y","theta"]
        csv_input = pd.concat([csv_input,data0_df])
    csv_input=csv_input.reset_index()

    #刺激開始時点の回旋を0にそろえる
    z=csv_input['theta'].values[stimustart]
    csv_input['theta']=csv_input['theta']-z
    
    z=csv_input['x'].values[stimustart]
    csv_input['x']=csv_input['x']-z
    
    z=csv_input['y'].values[stimustart]
    csv_input['y']=csv_input['y']-z


    csv_input.to_csv(allfolder + "/" +foldername+'/test7.csv')

    csv = csv_input[["num of Kp","x","y","theta"]]

    a_csv=csv['theta'].values
    b_csv=csv['x'].values
    c_csv=csv['y'].values
    list.append(a_csv)
    list2.append(b_csv)
    list3.append(c_csv)

    csv=csv.reset_index()
    csv.plot(subplots=True,y=['x','y','theta'])
    csv_stimulating0=csv[ stimustart: int(stimufinish+1)]
    popt0, pcov0 = curve_fit(fit_func, np.arange(len(csv_stimulating0)), csv_stimulating0['theta'].values, p0=(1, 0.25, 0))
    list0.append(abs(popt0[0]))
    
    fig,ax=plt.subplots(1,1,figsize=(15,10))
    ax.plot(np.arange(len(csv)),csv['theta'].values,label='theta')
    ax.set_xlabel('frame')
    ax.set_ylabel('amp[degree]')
    #ax[2].plot(np.arange(len(csv_total)), fit_func(np.array(np.arange(len(csv_total))), *popt), alpha=0.5, color="crimson")
    ax.plot(csv_stimulating0['index'], fit_func(np.array(np.arange(len(csv_stimulating0))), *popt0), alpha=0.5, color="crimson")
    ax.axvline(x=stimustart, color='red',  linewidth=3)#刺激開始
    ax.axvline(x=stimufinish, color='red',  linewidth=3)#刺激終了
    plt.savefig(allfolder+"/"+foldername+"/result1.png")

    
    csv_total = csv_total+csv
    count_file +=1
    csv_total.to_csv(allfolder + "/" +foldername+'/test_total7.csv')

list_=np.array(list)
mean=list_.mean(axis=0)
std=list_.std(axis=0)

list_=np.array(list2)
mean2=list_.mean(axis=0)
std2=list_.std(axis=0)

list_=np.array(list3)
mean3=list_.mean(axis=0)
std3=list_.std(axis=0)

stdev = statistics.stdev(list0)
mean0 = statistics.mean(list0)
print(list0)
print(mean0)
print(stdev)

# In[237]:


print(count_file)
csv_total = csv_total/count_file
csv_total=csv_total.reset_index()     
#csv_total.to_csv(allfolder + '/test.csv')
csv_total.plot(subplots=True,y=['x','y','theta'])
csv_stimulating=csv_total[ stimustart: int(stimufinish+1)]


fig,ax=plt.subplots(4,1,figsize=(15,10))

#popt, pcov = curve_fit(fit_func, x, y, p0=(0.5, 1.0, 100))
#popt, pcov = curve_fit(fit_func, np.arange(len(csv_total)), csv_total['theta'].values, p0=(1, 3, 0))

popt, pcov = curve_fit(fit_func, np.arange(len(csv_stimulating)), csv_stimulating['theta'].values, p0=(1, 0.25, 0))
popt2, pcov2 = curve_fit(fit_func, np.arange(len(csv_stimulating)), csv_stimulating['x'].values, p0=(1, 0.25, 0))
popt3, pcov3 = curve_fit(fit_func, np.arange(len(csv_stimulating)), csv_stimulating['y'].values, p0=(1, 0.25, 0))

#print(f"best-fit parameters = {popt}")
#print(f"covariance = \n{pcov}")

np.savetxt(allfolder+'/'+'popt.txt',popt)
peer = np.sqrt(np.diag(pcov))

d_a = round(peer[0],5)
d_b = round(peer[1],5)
d_c = round(peer[2],5)

print('curve_fittingの結果(theta)')
print('振幅:' + str(round(popt[0],4)) + '±' + str(d_a)+"(degree)")
print('周波数:' + str(round(popt[1],4)) + '±' + str(d_b)+"(Hz)")
print('位相:' + str(round(popt[2],4)*360/fps) + '±' + str(d_c*360/fps)+"(degree)")

np.savetxt(allfolder+'/'+'popt2.txt',popt2)
peer = np.sqrt(np.diag(pcov2))

d_a = round(peer[0],5)
d_b = round(peer[1],5)
d_c = round(peer[2],5)

print('curve_fittingの結果(x)')
print('振幅:' + str(round(popt2[0],4)) + '±' + str(d_a)+"(degree)")
print('周波数:' + str(round(popt2[1],4)) + '±' + str(d_b)+"(Hz)")
print('位相:' + str(round(popt2[2],4)*360/fps) + '±' + str(d_c*360/fps)+"(degree)")

np.savetxt(allfolder+'/'+'popt3.txt',popt3)
peer = np.sqrt(np.diag(pcov3))

d_a = round(peer[0],5)
d_b = round(peer[1],5)
d_c = round(peer[2],5)

print('curve_fittingの結果(y)')
print('振幅:' + str(round(popt3[0],4)) + '±' + str(d_a)+"(degree)")
print('周波数:' + str(round(popt3[1],4)) + '±' + str(d_b)+"(Hz)")
print('位相:' + str(round(popt3[2],4)*360/fps) + '±' + str(d_c*360/fps)+"(degree)")

#fig, p = plt.subplots(1, 1, sharex=True)
ax[0].plot(np.arange(len(csv_total)),csv_total['x'].values,label='x')
ax[0].set_xlabel('frame')
ax[0].set_ylabel('x px')
ax[0].axvline(x=stimustart, color='red',  linewidth=3)#刺激開始
ax[0].axvline(x=stimufinish, color='red',  linewidth=3)#刺激終了
ax[0].plot(csv_stimulating['index'], fit_func(np.array(np.arange(len(csv_stimulating))), *popt2), alpha=0.5, color="crimson")
ax[0].fill_between(np.arange(len(csv_total)),mean2+std2,mean2-std2,alpha=0.2,color="blue")

ax[1].plot(np.arange(len(csv_total)),csv_total['y'].values,label='y')
ax[1].set_xlabel('frame')
ax[1].set_ylabel('y px')
ax[1].axvline(x=stimustart, color='red',  linewidth=3)#刺激開始
ax[1].axvline(x=stimufinish, color='red',  linewidth=3)#刺激終了
ax[1].plot(csv_stimulating['index'], fit_func(np.array(np.arange(len(csv_stimulating))), *popt3), alpha=0.5, color="crimson")
ax[1].fill_between(np.arange(len(csv_total)),mean3+std3,mean3-std3,alpha=0.2,color="blue")

ax[2].plot(np.arange(len(csv_total)),csv_total['theta'].values,label='theta')
ax[2].set_xlabel('frame')
ax[2].set_ylabel('amp[degree]')
#ax[2].plot(np.arange(len(csv_total)), fit_func(np.array(np.arange(len(csv_total))), *popt), alpha=0.5, color="crimson")
ax[2].plot(csv_stimulating['index'], fit_func(np.array(np.arange(len(csv_stimulating))), *popt), alpha=0.5, color="crimson")
ax[2].axvline(x=stimustart, color='red',  linewidth=3)#刺激開始
ax[2].axvline(x=stimufinish, color='red',  linewidth=3)#刺激終了
ax[2].fill_between(np.arange(len(csv_total)),mean+std,mean-std,alpha=0.2,color="blue")
#ax[2].set_ylim(-0.05,0.05)

ax[3].plot(np.arange(len(csv_total)),csv_total['num of Kp'].values,label='Kp')
ax[3].set_xlabel('frame')
ax[3].set_ylabel('num of Kp')
#ax[3].axvline(x=30, color='red',  linewidth=3)
#ax[3].axvline(x=(len(a)-stimustart), color='red',  linewidth=3)
#ax[3].axhline(y=100, color='red',  linewidth=1)
#ax[3].axhline(y=200, color='red',  linewidth=1)
#ax[3].axhline(y=300, color='red',  linewidth=1)
#ax[3].set_ylim(0, 500)
plt.savefig(allfolder+"/result.png")
#plt.show()
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




