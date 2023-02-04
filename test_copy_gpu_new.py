from tracking_torison_variable_frame_distance_gpu import *
from def_doukousuitei import*

import pandas as pd

moviename="murata_gannkyu_test4.mp4"

frame_distance1=1
video_path = './movie/0201/'+moviename
save_file_path1='./test_fd1/0201/'+moviename+'/kaiseki_test'
video = cv2.VideoCapture(video_path)
print('OpenVideo:', video.isOpened())
#roi = [200,0,100,1000]
#roi2 = [200,200,300,300]
#eye_center=[1000,298]
df=pd.DataFrame()
#fase1 画面上の瞳孔中心を得る

eye_center_x,eye_center_y,high,wide=doukousuitei(video, save_file_path=save_file_path1,startframe=240,time=30)

#fase2 瞳孔中心から回転中心の長さRを得る

eye_center_x_r,eye_center_y_r,high,wide=doukousuitei(video, save_file_path=save_file_path1,startframe=218,time=10)

r=math.sqrt((eye_center_x_r-eye_center_x)**2+(eye_center_y_r-eye_center_y)**2)

z=wide/high

if z>1:

    z=1/z


R=r/math.sqrt(1-((z)**2))

#fase3 眼球の内、二つの範囲の並進量を得る

df_test,roi1,roi_lumi,stim_start,stim_finish=tracking_eye(video, save_file_path=save_file_path1, crop_mode='manual-ROI', iris_r=50,
                                match_type='best', plot_status=False,norm_hist=False,frame_distance=frame_distance1)

df_test2,roi2,roi_lumi,stim_start,stim_finish=tracking_eye(video, save_file_path=save_file_path1, crop_mode='manual-ROI2', iris_r=50,
                                match_type='best', plot_status=False,ROI2=roi_lumi,norm_hist=False,frame_distance=frame_distance1)


#df_test,roi,roi2,stim_start,stim_finish=tracking_eye(video, save_file_path=save_file_path1, crop_mode='manual-ROI', iris_r=50,match_type='best', plot_status=False,norm_hist=False,frame_distance=frame_distance1)                                
video.release()

obj={"roi1":roi1,"roi2":roi2,"roi_lumi":roi_lumi,"R":R}
with open(save_file_path1+'conditons_.yaml', 'w') as file:
      yaml.dump(obj, file)

df_test = df_test.rename(columns={"dx":"dx1","dy":"dy1"})
df_test2=df_test2.rename(columns={"dx":"dx2","dy":"dy2"})
df_test.to_csv(save_file_path1+"df_test.csv")
df_test2.to_csv(save_file_path1+"df_test2.csv")
df_dx_dy=pd.concat([df_test[["dx1","dy1"]],df_test2[["dx2","dy2"]]],axis=1)

df_dx_dy.to_csv(save_file_path1+"dx_dy.csv")


#fase4 二つの範囲の並進量から、回転軸ｎとθを得る

xc1=roi1[0]+roi1[2]/2
yc1=roi1[1]+roi1[3]/2
xc2=roi2[0]+roi2[2]/2
yc2=roi2[1]+roi2[3]/2

for i in range(len(df_dx_dy)):
    #print(i)
    dx1=df_dx_dy["dx1"][i]
    dy1=df_dx_dy['dy1'][i]
    x1=xc1-eye_center_x
    y1=yc1-eye_center_y
    z1= math.sqrt(R**2 - x1**2 - y1**2)
    p1=np.array([x1,y1,z1])

    n1=np.array([dx1,dy1,math.sqrt(R**2 - (x1+dx1)**2 - (y1+dy1)**2) - z1])

    dx2=df_dx_dy["dx2"][i]
    dy2=df_dx_dy['dy2'][i]
    x2=xc2-eye_center_x
    y2=yc2-eye_center_y
    z2=math.sqrt(R**2 - x2**2 - y2**2)
    n2=np.array([dx1,dy1,math.sqrt(R**2 - (x2+dx2)**2 - (y2+dy2)**2) - z2])

    n_axis= np.cross(n1,n2)

    s = 1/math.sqrt(n_axis[0]**2+n_axis[1]**2+n_axis[2]**2)
    
    n_axis=n_axis*s 
    #print('n_axis')
    #print(n_axis)
    a1 = np.dot(n_axis,p1)
    n_axis1=a1*n_axis
    p1_=np.array([x1+dx1,y1+dy1,math.sqrt(R**2 - (x1+dx1)**2 - (y1+dy1)**2)])
    a2 = np.dot(n_axis,p1_)
    n_axis2=a2*n_axis
    z=np.dot((p1-n_axis1),(p1_-n_axis2))/(math.sqrt(np.dot((p1-n_axis1),(p1-n_axis1)))*math.sqrt(np.dot((p1_-n_axis2),(p1_-n_axis2))))
    if z>1:
        z=1

    theta = math.acos(z)*180/math.pi
    #print('theta')
    #print(theta)
    #print(theta)
    #print(n_axis)
    p1xy=p1-n_axis1
    p1xy[2]=0
    
    p1_xy=p1_-n_axis2
    p1_xy[2]=0
    
    z=np.dot((p1xy),(p1_xy))/(math.sqrt(np.dot((p1xy),(p1xy)))*math.sqrt(np.dot((p1_xy),(p1_xy))))
    if z>1:
        z=1
    
    xytheta=math.acos(z)*180/math.pi
    #print('xytheta')
    #print(xytheta)

    p1yz=p1-n_axis1
    p1yz[0]=0
    p1_yz=p1_-n_axis2
    p1_yz[0]=0

    z=np.dot((p1yz),(p1_yz))/(math.sqrt(np.dot((p1yz),(p1yz)))*math.sqrt(np.dot((p1_yz),(p1_yz))))
    if z>1:
        z=1


    yztheta=math.acos(z)*180/math.pi
    #print('yztheta')
    #print(yztheta)

    p1zx=p1-n_axis1
    p1zx[1]=0
    p1_zx=p1_-n_axis2
    p1_zx[1]=0

    z=np.dot((p1zx),(p1_zx))/(math.sqrt(np.dot((p1zx),(p1zx)))*math.sqrt(np.dot((p1_zx),(p1_zx))))
    if z>1:
        z=1
       
    zxtheta=math.acos(z)*180/math.pi
    #print('zxtheta')
    #print(zxtheta)

    df_=pd.DataFrame([{'n_axis_x':n_axis[0],'n_axis_y':n_axis[1],'n_axis_z':n_axis[2],
                                  'dtheta':theta,'xydtheta':xytheta,'yzdtheta':yztheta,
                                  'zxdtheta':zxtheta  }])

    df=pd.concat((df,df_))

df = df.reset_index()
df['xytheta'] = np.cumsum(df['xydtheta'].values)
df['yztheta'] = np.cumsum(df['yzdtheta'].values)
df['zxtheta'] = np.cumsum(df['zxdtheta'].values)
df['theta'] = np.cumsum(df['dtheta'].values)    

df.to_csv(save_file_path1+'result_tracking_axis_angle.csv')
    
df.plot(subplots=True,y=['xytheta','yztheta','zxtheta','theta','xydtheta','yzdtheta','zxdtheta','dtheta'])
fig,ax=plt.subplots(3,2,figsize=(15,10))
ax[0,0].plot(np.arange(len(df)),df['yztheta'].values,label='X')
ax[0,0].set_xlabel('frame')
ax[0,0].set_ylabel('X')
ax[1,0].plot(np.arange(len(df)),df['zxtheta'].values,label='Y')
ax[1,0].set_xlabel('frame')
ax[1,0].set_ylabel('Y')
ax[2,0].plot(np.arange(len(df)),df['xytheta'].values,label='Z')
ax[2,0].set_xlabel('frame')
ax[2,0].set_ylabel('Z')
#ax[2].grid()
#ax[2,0].set_ylim(-2, 2)
#ax[3,0].plot(np.arange(len(df)),df['theta'].values,label='theta')
#ax[3,0].set_xlabel('frame')
#ax[3,0].set_ylabel('theta')

ax[0,1].scatter(df['yzdtheta'].values,df['zxdtheta'].values,label='XY',s=10)
ax[0,1].set_xlabel('X axis')
ax[0,1].set_ylabel('Y axis')
ax[0,1].set_xlim(-2, 2)
ax[0,1].set_ylim(-2, 2)
ax[1,1].scatter(df['yzdtheta'].values,df['xydtheta'].values,label='XZ',s=10)
ax[1,1].set_xlabel('X axis')
ax[1,1].set_ylabel('Z axis')
ax[1,1].set_xlim(-2, 2)
ax[1,1].set_ylim(-2, 2)
ax[2,1].scatter(df['zxdtheta'].values,df['xydtheta'].values,label='YZ',s=10)
ax[2,1].set_xlabel('Y axis')
ax[2,1].set_ylabel('Z axis')
ax[2,1].set_xlim(-2, 2)
ax[2,1].set_ylim(-2, 2)
#ax[2].grid()
#ax[2,0].set_ylim(-2, 2)
#ax[3,1].plot(np.arange(len(df)),df_test['theta'].values,label='theta')
#ax[3,1].set_xlabel('frame')
#ax[3,1].set_ylabel('theta')   
plt.show()
exit()







"""
frame_distance2=1
video_path = './movie/test-movie8/'+moviename
save_file_path2='./test_fd1/test-movie8/'+moviename+'/kaiseki_test_gpu_500'
video = cv2.VideoCapture(video_path)
print('OpenVideo:', video.isOpened())                                
df_test2,roi,roi2,stim_start,stim_finish=tracking_eye(video, save_file_path=save_file_path2, crop_mode='fix_ROI', iris_r=50,
                                match_type='best', plot_status=False,ROI=roi,ROI2=roi2,norm_hist=False,frame_distance=frame_distance2)
video.release()

print(frame_distance1)
print('kength of files df',len(df_test))
print(frame_distance2)
print('kength of files df2',len(df_test2))
"""
#可視化
df_test.plot(subplots=True,y=['x','y','theta','dx','dy','dtheta'])
fig,ax=plt.subplots(4,2,figsize=(15,10))
ax[0,0].plot(np.arange(len(df_test)),df_test['x'].values,label='x')
ax[0,0].set_xlabel('frame')
ax[0,0].set_ylabel('x ')
ax[0,0].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
ax[0,0].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
ax[1,0].plot(np.arange(len(df_test)),df_test['y'].values,label='y')
ax[1,0].set_xlabel('frame')
ax[1,0].set_ylabel('y px')
ax[1,0].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
ax[1,0].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
ax[2,0].plot(np.arange(len(df_test)),df_test['theta'].values,label='theta')
ax[2,0].set_xlabel('frame')
ax[2,0].set_ylabel('amp[degree]')
#ax[2].grid()
ax[2,0].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
ax[2,0].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
#ax[2,0].set_ylim(-2, 2)
ax[3,0].plot(np.arange(len(df_test)),df_test['num of Kp'].values,label='Kp')
ax[3,0].set_xlabel('frame')
ax[3,0].set_ylabel('num of Kp')
ax[3,0].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
ax[3,0].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
ax[3,0].axhline(y=100, color='red',  linewidth=1)
ax[3,0].axhline(y=200, color='red',  linewidth=1)
ax[3,0].axhline(y=300, color='red',  linewidth=1)
#ax[3].set_ylim(0, 500)

ax[0,1].plot(np.arange(len(df_test)),df_test['dx'].values,label='dx')
ax[0,1].set_xlabel('frame')
ax[0,1].set_ylabel('dx ')
#ax[0].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
#ax[0].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
ax[1,1].plot(np.arange(len(df_test)),df_test['dy'].values,label='dy')
ax[1,1].set_xlabel('frame')
ax[1,1].set_ylabel('dy px')
#ax[1].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
#ax[1].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
ax[2,1].plot(np.arange(len(df_test)),df_test['dtheta'].values,label='dtheta')
ax[2,1].set_xlabel('frame')
ax[2,1].set_ylabel('dtheta')
#ax[2,1].plot(np.arange(len(df)),data_lofilt_dt,'r',label='theta')
ax[2,1].grid()
#ax[2].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
#ax[2].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
#ax[2,1].set_ylim(-1.5, 1.5)
ax[3,1].plot(np.arange(len(df_test)),df_test['num of Kp'].values,label='Kp')
ax[3,1].set_xlabel('frame')
ax[3,1].set_ylabel('num of Kp')
#ax[3].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
#ax[3].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
ax[3,1].axhline(y=100, color='red',  linewidth=1)
ax[3,1].axhline(y=200, color='red',  linewidth=1)
ax[3,1].axhline(y=300, color='red',  linewidth=1)
#for a in ax:
  #a.grid()
plt.savefig(save_file_path1+"/figure2")

"""
df_test2.plot(subplots=True,y=['x','y','theta','dx','dy','dtheta'])
fig2,ax2=plt.subplots(4,2,figsize=(15,10))
ax2[0,0].plot(np.arange(len(df_test2)),df_test2['x'].values,label='x')
ax2[0,0].set_xlabel('frame')
ax2[0,0].set_ylabel('x px')
ax2[0,0].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
ax2[0,0].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
#ax2[0].set_ylim(-1000, 1000)
ax2[1,0].plot(np.arange(len(df_test2)),df_test2['y'].values,label='y')
ax2[1,0].set_xlabel('frame')
ax2[1,0].set_ylabel('y px')
ax2[1,0].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
ax2[1,0].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
#ax2[1].set_ylim(-1000, 1000)
ax2[2,0].plot(np.arange(len(df_test2)),df_test2['theta'].values,label='theta')
ax2[2,0].set_xlabel('frame')
ax2[2,0].set_ylabel('amp[degree]')
ax2[2,0].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
ax2[2,0].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
ax2[2,0].set_ylim(-2, 2)
ax2[3,0].plot(np.arange(len(df_test2)),df_test2['num of Kp'].values,label='Kp')
ax2[3,0].set_xlabel('frame')
ax2[3,0].set_ylabel('num of kp')
ax2[3,0].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
ax2[3,0].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
ax2[3,0].axhline(y=100, color='red',  linewidth=1)
ax2[3,0].axhline(y=200, color='red',  linewidth=1)
ax2[3,0].axhline(y=300, color='red',  linewidth=1)
#ax2[3].set_ylim(0, 500)

ax2[0,1].plot(np.arange(len(df_test)),df_test['dx'].values,label='dx')
ax2[0,1].set_xlabel('frame')
ax2[0,1].set_ylabel('dx ')
#ax2[0].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
#ax2[0].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
ax2[1,1].plot(np.arange(len(df_test)),df_test['dy'].values,label='dy')
ax2[1,1].set_xlabel('frame')
ax2[1,1].set_ylabel('dy px')
#ax2[1].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
#ax2[1].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
ax2[2,1].plot(np.arange(len(df_test)),df_test['dtheta'].values,label='dtheta')
ax2[2,1].set_xlabel('frame')
ax2[2,1].set_ylabel('dtheta')
#ax2[2,1].plot(np.arange(len(df)),data_lofilt_dt,'r',label='theta')
ax2[2,1].grid()
#ax2[2].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
#ax2[2].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
#ax2[2,1].set_ylim(-1.5, 1.5)
ax2[3,1].plot(np.arange(len(df_test)),df_test['num of Kp'].values,label='Kp')
ax2[3,1].set_xlabel('frame')
ax2[3,1].set_ylabel('num of Kp')
#ax2[3].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
#ax2[3].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
ax2[3,1].axhline(y=100, color='red',  linewidth=1)
ax2[3,1].axhline(y=200, color='red',  linewidth=1)
ax2[3,1].axhline(y=300, color='red',  linewidth=1)
"""
"""
for a in ax2:
  a.grid()
plt.savefig(save_file_path2+"/figure4")
"""
plt.show()