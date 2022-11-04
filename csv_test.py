import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

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
fig,ax=plt.subplots(4,1,figsize=(15,10))
ax[0].plot(np.arange(len(csv_total)),csv_total['x'].values,label='x')
ax[0].set_xlabel('frame')
ax[0].set_ylabel('x ')
ax[0].axvline(x=30, color='red',  linewidth=3)
ax[0].axvline(x=(len(csv_total)-90), color='red',  linewidth=3)
ax[1].plot(np.arange(len(csv_total)),csv_total['y'].values,label='y')
ax[1].set_xlabel('frame')
ax[1].set_ylabel('y px')
ax[1].axvline(x=30, color='red',  linewidth=3)
ax[1].axvline(x=(len(csv_total)-90), color='red',  linewidth=3)
ax[2].plot(np.arange(len(csv_total)),csv_total['theta'].values,label='theta')
ax[2].set_xlabel('frame')
ax[2].set_ylabel('amp[degree]')
ax[2].axvline(x=30, color='red',  linewidth=3)
ax[2].axvline(x=(len(csv_total)-90), color='red',  linewidth=3)
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
 