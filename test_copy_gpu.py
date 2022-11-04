from tracking_torison_variable_frame_distance_gpu import *

file = "1025_"
path = "./movie/"+file
files = os.listdir(path)
filename="csvtest_gankyuitibu_gpu7"

ROIFRAG=True

for moviename in files:

  frame_distance1=10
  video_path = path + "/" +moviename
  #video_path = './movie/hara/20221005AH左右方向GVS中視点変化_ON2.mp4'
  save_file_path1 = './test_fd10/'+file+'/'+moviename+'/'+filename
  #save_file_path1='./test_fd10/hara/20221005AH左右方向GVS中視点変化_ON2.mp4/gankyuzentai3_gpu3'
  video = cv2.VideoCapture(video_path)
  print('OpenVideo:', video.isOpened())
  print(moviename)

  if ROIFRAG ==False:
    
    df_test,roi,roi2,stim_start,stim_finish=tracking_eye(video, save_file_path=save_file_path1, crop_mode='fix_ROI', iris_r=50,
                                match_type='best', plot_status=False,ROI=roi,ROI2=roi2,norm_hist=False,frame_distance=frame_distance1)
    
  if ROIFRAG:
    
    df_test,roi,roi2,stim_start,stim_finish=tracking_eye(video, save_file_path=save_file_path1, crop_mode='manual-ROI', iris_r=50,
                                match_type='best', plot_status=False,norm_hist=False,frame_distance=frame_distance1)
                                
    ROIFRAG = False
  
                        
  video.release()

  frame_distance2=1
  video_path = path + "/" +moviename
  #video_path = './movie/hara/20221005AH左右方向GVS中視点変化_ON2.mp4'
  save_file_path2 = './test_fd1/'+file+'/'+moviename+'/'+filename
  #save_file_path2='./test_fd1/hara/20221005AH左右方向GVS中視点変化_ON2.mp4/gankyuzentai3_gpu3'
  video = cv2.VideoCapture(video_path)
  print('OpenVideo:', video.isOpened())    
                            
  df_test2,roi,roi2,stim_start,stim_finish=tracking_eye(video, save_file_path=save_file_path2, crop_mode='fix_ROI', iris_r=50,
                                match_type='best', plot_status=False,ROI=roi,ROI2=roi2,norm_hist=False,frame_distance=frame_distance2)
  video.release()

  print(frame_distance1)
  print('kength of files df',len(df_test))
  print(frame_distance2)
  print('kength of files df2',len(df_test2))

  #可視化
  df_test.plot(subplots=True,y=['x','y','theta'])
  fig,ax=plt.subplots(4,1,figsize=(15,10))
  ax[0].plot(np.arange(len(df_test)),df_test['x'].values,label='x')
  ax[0].set_xlabel('frame')
  ax[0].set_ylabel('x ')
  #ax[0].axvline(x=30, color='red',  linewidth=3)
  #ax[0].axvline(x=(len(df_test)-90), color='red',  linewidth=3)
  ax[1].plot(np.arange(len(df_test)),df_test['y'].values,label='y')
  ax[1].set_xlabel('frame')
  ax[1].set_ylabel('y px')
  ax[1].axvline(x=30, color='red',  linewidth=3)
  ax[1].axvline(x=(len(df_test)-90), color='red',  linewidth=3)
  ax[2].plot(np.arange(len(df_test)),df_test['theta'].values,label='theta')
  ax[2].set_xlabel('frame')
  ax[2].set_ylabel('amp[degree]')
  ax[2].axvline(x=30, color='red',  linewidth=3)
  ax[2].axvline(x=(len(df_test)-90), color='red',  linewidth=3)
  ax[3].plot(np.arange(len(df_test)),df_test['num of Kp'].values,label='Kp')
  ax[3].set_xlabel('frame')
  ax[3].set_ylabel('num of Kp')
  #ax[3].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
  #ax[3].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
  ax[3].axhline(y=100, color='red',  linewidth=1)
  ax[3].axhline(y=200, color='red',  linewidth=1)
  ax[3].axhline(y=300, color='red',  linewidth=1)
  #ax[3].set_ylim(0, 500)
  for a in ax:
    a.grid()
  plt.savefig(save_file_path1+"/figure2")

  df_test2.plot(subplots=True,y=['x','y','theta'])
  fig2,ax2=plt.subplots(4,1,figsize=(15,10))
  ax2[0].plot(np.arange(len(df_test2)),df_test2['x'].values,label='x')
  ax2[0].set_xlabel('frame')
  ax2[0].set_ylabel('x px')
  ax2[0].axvline(x=30, color='red',  linewidth=3)
  ax2[0].axvline(x=(len(df_test2)-90), color='red',  linewidth=3)
  #ax2[0].set_ylim(-1000, 1000)
  ax2[1].plot(np.arange(len(df_test2)),df_test2['y'].values,label='y')
  ax2[1].set_xlabel('frame')
  ax2[1].set_ylabel('y px')
  ax2[1].axvline(x=30, color='red',  linewidth=3)
  ax2[1].axvline(x=(len(df_test2)-90), color='red',  linewidth=3)
  #ax2[1].set_ylim(-1000, 1000)
  ax2[2].plot(np.arange(len(df_test2)),df_test2['theta'].values,label='theta')
  ax2[2].set_xlabel('frame')
  ax2[2].set_ylabel('amp[degree]')
  ax2[2].axvline(x=30, color='red',  linewidth=3)
  ax2[2].axvline(x=(len(df_test2)-90), color='red',  linewidth=3)
  #ax2[2].set_ylim(-1.5, 1.5)
  ax2[3].plot(np.arange(len(df_test2)),df_test2['num of Kp'].values,label='Kp')
  ax2[3].set_xlabel('frame')
  ax2[3].set_ylabel('num of kp')
  #ax2[3].axvline(x=stim_start, color='red',  linewidth=3)#刺激開始
  #ax2[3].axvline(x=stim_finish, color='red',  linewidth=3)#刺激終了
  ax2[3].axhline(y=100, color='red',  linewidth=1)
  ax2[3].axhline(y=200, color='red',  linewidth=1)
  ax2[3].axhline(y=300, color='red',  linewidth=1)
  #ax2[3].set_ylim(0, 500)


  for a in ax2:
    a.grid()
  plt.savefig(save_file_path2+"/figure4")

  #plt.show()