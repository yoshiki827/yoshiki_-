'''
- Developed by Akiyoshi Hara(2021/11)

# 調整の必要なパラメータ
- threshold: 特徴量点のマッチングを行う場合に距離としての誤対応を排除するためのパラメータ
- 画像前処理の最適化(そもそもなぜ最適化がいるのかの問題)
- 距離尺度の設定 初期はL2ノルム
- SURF特徴量の検出パラメータの設定
  >Parameters:
  >hessianThreshold – Threshold for hessian keypoint detector used in SURF.
  >nOctaves – Number of pyramid octaves the keypoint detector will use.
  >nOctaveLayers – Number of octave layers within each octave.
  >extended – Extended descriptor flag(true - use extended 128-element   descriptors
  > false - use 64-element descriptors).
  >upright – Up-right or rotated features flag(true - do not compute orientation   of > features
  false - compute orientation).

# TODO
- 精度の確認
 - 補正を入れるべきか否か検討する
   - 中心については補正を入れたほうがいいことが判明
  - ヒストグラムで誤差を表現

# 追加機能(できたらすること)
- トラッキング過程を動画として確認できるようにする
- 解像度の大きい動画では計算に時間がかかるので，対策を練る
- 回転中心の移動軌跡の計算
 - 途中まで実装したが，これを使う予定もないのでひとまずペンディングしておく

# 参考文献
- https: //jp.mathworks.com/help/vision/ref/detectsurffeatures.html
- https://docs.opencv.org/3.0-beta/modules/xfeatures2d/doc/nonfree_features.html?highlight =cv2.surf#cv2.SURF
-
'''

# surfの実装
from sys import flags
from this import d
from PIL import Image
import yaml
import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import math
import seaborn as sns
sns.set_style("darkgrid")


def tracking_eye(video, save_file_path='./test', crop_mode='manual_ROI', match_type='best',
                 iris_r=0, plot_status=True, show_correspondence_points=True, ROI=None,ROI2=None, 
                 circle=None, norm_hist=True,frame_distance=1):
  '''
  実際にトラッキングをする関数，隣り合う画像間での変化の追従を行い、
  その積分として現在の位置を計算する
  video: 解析対象とするビデオオブジェクト
  save_file_path='./test': 実際にトラッキングをして保存する先のパス
  crop_mode='manual_ROI': (manual-circle, manual-ROI, auto_eye, fix_ROI')トラッキングの対象となる領域の取り出し方
  match_type='best' : (best, knn)対応点計測を行う際のモード
  iris_r=0: 自動的に黒目抽出を行う際に，その外に半径方向どのくらい大きく取るかを決定する
  plot_status=True: 各処理の初期条件を可視化して確認する
  show_correspondence_points=False: トラッキングの対応関係を可視化する(デバッグ用)
  ROI=None: crop_mode='fix_ROI'の場合に，どの領域を取りたいかを指定する．
  ROI[x1,y1,x2,y1]は(x1, y1)~(x1+x2, y1+y2)として指定
  circle=None: crop_mode='fix_circle'の場合に，どの領域を取りたいかを指定する．
  norm_hist=True:画像のヒストグラム平滑化を使用するかどうか
  frame_distance=1:計算するフレームの距離を定義（フレーム数で定義する）
  '''
  pro_bar_num = 10  # 全体の進捗をなん分割して表示するか
  df = pd.DataFrame()
  df_csv = pd.DataFrame()
  os.makedirs(save_file_path, exist_ok=True)
  save_file_path = save_file_path+'/'
  # video infomation
  video.set(cv2.CAP_PROP_POS_FRAMES,30*5)#動画のスタート位置
  total_frame_num =30*2#video.get(cv2.CAP_PROP_FRAME_COUNT) 

  v_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
  v_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
  print('total_frame_num', total_frame_num,
        'v_width', v_width, 'v_height', v_height)

  q_kps=[] #フレームの履歴を保存するキュー
  q_deses = []

  #トラッキング・マッチング用のクラスの定義
  # double hessianThreshold、int nOctaves = 4、int nOctaveLayers = 2、bool extended = true、bool upright = false ）
  surf0 = cv2.xfeatures2d.SURF_create(
      hessianThreshold=1, nOctaves=3, nOctaveLayers=6)
  surf = cv2.cuda.SURF_CUDA_create(_hessianThreshold=1, _nOctaves=3, _nOctaveLayers=6)
  bf = cv2.BFMatcher(crossCheck=True)  # 距離尺度の設定 初期はL2ノルム
  #bf = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)
  # マッチング関数の定義
  if match_type == 'knn':
    match_func = match_knn
  if match_type == 'best':
    match_func = match_best

  frame_past = cv2.cuda_GpuMat()
  frame_current = cv2.cuda_GpuMat()
  img_kp_gpu = cv2.cuda_GpuMat()

  # frame_distanceだけキューに追加する処理
  crop_flag=0 #0：初期、1:fix_circle,2:fix_ROI
  for _ in range(frame_distance):
    # フレームの読み込み
    ret, frame_past0 = video.read()
    frame_past.upload(frame_past0)
    # pre-image process
    frame_past = pre_image_process(
        frame_past, show_state=plot_status, norm_hist=norm_hist,save_file_path = save_file_path)
    frame_past0= frame_past.download()
    # calc kp in all area
    kp_past, des_past = surf0.detectAndCompute(frame_past0, None)
    #kp_past_g, des_past_g = surf.detectWithDescriptors(frame_past, None)
    # select kp in area
    if (crop_mode == 'manual-circle') & (crop_flag==0):
      ROI, circle = crop_ROI(frame_past)
      kp_past, des_past = select_kps_in_area(frame_past0, kp_past, des_past, circle,
                                            save_file_path, show_state=plot_status)
      crop_flag=1
    if (crop_mode == 'auto_eye') & (crop_flag==0):
      circle = detect_iris(frame_past0, save_file_path, show_state=plot_status)
      ROI = calc_ROI(circle)
      kp_past, des_past = select_kps_in_area(frame_past0, kp_past, des_past, [circle[0], circle[1], circle[2]+iris_r],
                                            save_file_path, show_state=plot_status)
      crop_flag=1
    if (crop_mode == 'manual-ROI') & (crop_flag==0):
      ROI, circle = crop_ROI(frame_past0)
      #FRAGLEDのROI
      ROI2,circle2, = crop_ROI(frame_past0) 
      kp_past, des_past = select_kps_in_area(frame_past0, kp_past, des_past, ROI,
                                            save_file_path, area_type='ROI', show_state=plot_status)
      crop_flag=2
    if (crop_mode == 'fix-circle') | (crop_flag==1):
      ROI = calc_ROI(circle)
      kp_past, des_past = select_kps_in_area(frame_past0, kp_past, des_past, circle,
                                            save_file_path, show_state=plot_status)
    if (crop_mode == 'fix_ROI') | (crop_flag==2):
      circle = calc_circle(ROI)
      kp_past, des_past = select_kps_in_area(frame_past0, kp_past, des_past, ROI,
                                            save_file_path, area_type='ROI', show_state=plot_status)
    q_kps.append(kp_past)
    q_deses.append(des_past)

  print('Num of Kp:', len(q_kps[0]))
  print('save conditions')
  obj = {'crop_mode': crop_mode, 'circle': circle, 'ROI': ROI, 'match_type': match_type,
         'iris_r': iris_r, 'plot_status': plot_status, 'show_correspondence_points': show_correspondence_points}
  with open(save_file_path+'conditons.yaml', 'w') as file:
      yaml.dump(obj, file)

  print('start processing')
  start = time.time()
  """
  print(type(q_kps))
  print(type(q_deses))
  print(type(kp_past))
  print(type(des_past))
  """
  

  

  for _ in range(int(total_frame_num)-frame_distance):
    #キューの戦闘を取り出す
    kp_past = q_kps.pop(0)
    des_past = q_deses.pop(0)
    
    #最新のフレームに情報を更新
    ret, frame_current0 = video.read()

    #画像を切り抜き、FRAGLEDの輝度を保存する
    df_csv =luminance_check(frame_current0,ROI=ROI2,df=df_csv,framenum=_)


    frame_current.upload(frame_current0)
    frame_current = pre_image_process(
        frame_current, show_state=False, norm_hist=norm_hist)
    frame_current0 = frame_current.download()    

    #特徴量の計算
    #kp_current, des_current = surf.detectAndCompute(frame_current, None)
    kp_current, des_current = surf.detectWithDescriptors(frame_current, None)
    
    des_current_cpu = des_current.download()
    #kp_current_cpu = kp_current.download()
    kp_current_cpu = cv2.cuda_SURF_CUDA.downloadKeypoints(surf, kp_current)
    #print(type(kp_current_cpu))
    #print(type(des_current_cpu))
    #print(kp_current_cpu)
    #kp_current_cpu2 = tuple([tuple(e) for e in kp_current_cpu])
    #print(result)
    #print(type(kp_current_cpu))
    #print(kp_current_cpu)
    # 対応点の計算
    good, good_train_ind = match_func(bf, des_past, des_current_cpu)
    #print(type(good))
    #print(good)
    # アフィン変換行列の導出
    df = affine_mat(df, good, kp_past, kp_current_cpu,frame_distance)

   
    #__df=pd.DataFrame([{'index':(framenum)}])
    
    

    

    #古いkpを入れ替え，新しいkpを殻にする処理
    kp_past, des_past = [], []
    for i in good_train_ind:
      kp_past.append(kp_current_cpu[i])
      des_past.append(des_current_cpu[i])
    des_past = np.array(des_past)
    #des_past = cp.array(des_past)
    frame_past = frame_current0
    q_kps.append(kp_past)
    q_deses.append(des_past)

    """if show_correspondence_points:
        visualization_of_correspondence_points(
            match_type, frame_past, kp_past, frame_current, kp_current, good, save_file_path,_)"""
            
    ##現在の進捗の可視化
    if _ % int((total_frame_num-frame_distance)/pro_bar_num) == 0:
      show_progress_bar((total_frame_num-frame_distance), pro_bar_num, _, start)
      # 対応点の可視化
      if show_correspondence_points:
        visualization_of_correspondence_points(
            match_type, frame_past, kp_past, frame_current0, kp_current_cpu, good, save_file_path,_)
         
  df = df.reset_index()
  df['x'] = np.cumsum(df['dx'].values)
  df['y'] = np.cumsum(df['dy'].values)
  df['theta'] = np.cumsum(df['dtheta'].values)

  df.to_csv(save_file_path+'result_tracking.csv')

  #輝度変化が大きなframeを見つける(刺激開始と刺激終了)
  df_csv=df_csv.diff()
  df_csv=df_csv.abs()
  df_csv=df_csv.reset_index(drop=True)
  df_csv=df_csv.sort_values('luminance',ascending=False)
  df_csv=df_csv.reset_index()
  
  num_first = df_csv.at[0,'index']
  num_second = df_csv.at[1,'index']
  num_first_int=num_first.item()
  num_second_int = num_second.item()

  df_csv.to_csv(save_file_path+'result_tracking_luminance.csv')
  
  return df,ROI,ROI2,num_first_int,num_second_int

def select_kps_in_area(frame, kps1, des1, area, save_file_path, area_type='ROI',show_state=True):
  '''
  与えたれた円(ここでは黒目に相当)の周りに`iris_r`だけ半径を伸ばした円領域にあるkey pointを抽出する
  frame: 対象となる画像
  kps1: frameに対して計算されたkey point 
  des1: frameに対して計算されたdes
  area: 対象となるkey pointのある領域
  save_file_path: 対象となる動画の名前
  show_state=False: 可視化した結果を表示するかどうか
  '''
  
  kps_in_iris,des_in_iris=[],[]
  #kps_in_iris_gpu = cp.array(kps_in_iris)
  #des_in_iris_gpu = cp.array(des_in_iris)
  #area_gpu = cp.array(area)
  #print(type(kps1))
  """
  kps_c = kps1.download()
  kps_cpu=tuple(map(tuple, kps_c))
  print(type(kps_cpu))
  des_cpu = des1.download()
  print(type(des_cpu))
  """
  if area_type=='circle':
    for k,d in zip(kps1,des1):
      if (k.pt[0]-area[0])**2+(k.pt[1]-area[1])**2<(area[2])**2:
        kps_in_iris.append(k)
        des_in_iris.append(d)
  if area_type=='ROI':
    for k,d in zip(kps1,des1):
      if (k.pt[0]>=area[0]) &(k.pt[0]<=area[0]+area[2]) &(k.pt[1]>=area[1]) &(k.pt[1]<=area[1]+area[3]) :
        kps_in_iris.append(k)
        des_in_iris.append(d)  
  img_kp = cv2.drawKeypoints(frame, kps_in_iris, None)
  #img_kp = img_kp_gpu.download()
  cv2.imwrite(save_file_path+'img_surf.png', img_kp)
  #kps_in_iris =kps_in_iris_gpu.download()
  #des_in_iris =des_in_iris_gpu.download()
  if show_state:
    print('view')
    cv2.imshow('img_surf', img_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  return kps_in_iris, np.array(des_in_iris)

def detect_iris(frame, save_file_path,show_state=False):
  '''
  黒目の領域を抽出して画像として保存
  (主に使用しない方針になったので，あまり調整していない)
  frame: 対象となるフレーム
  save_file_path: 画像の保存先
  show_state=False: 計算結果の画像を表示するかどうか 
  '''
  # ___,frame__=cv2.threshold(frame,127,255,cv2.THRESH_BINARY)
  frame__=frame
  frame__=cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) # 画像の2値化
  circles = cv2.HoughCircles(frame__,cv2.HOUGH_GRADIENT,0.3,1000,
                            param1=100,param2=20,minRadius=0,maxRadius=200)  
  circles = np.uint16(np.around(circles))
  for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(frame__,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(frame__,(i[0],i[1]),2,(0,0,255),3)
  cv2.imwrite(save_file_path+"detected_circles.png", frame__)
  if show_state:
    cv2.imshow('detected circles', frame__)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  if len(circles)!=1:
    print('抽出された領域が1つでないので終了します')
    exit(-1)
  circle = circles[0,:][0]
  return circle

def crop_ROI(frame):
  '''
  対象領域の切り出し
  切り出した領域の内接円とROIを返す
  ROIは(x1, y1)~(x1+x2, y1+y2)の領域にあるセルになっていることに注意
  frame: 対象とするフレーム 
  '''
  #frame = frame.download()
  cv2.namedWindow('Select ROIs',2)
  ROI = cv2.selectROI('Select ROIs', frame, fromCenter = False, showCrosshair = False)
  circle=calc_circle(ROI)
  x1, y1, x2, y2 = ROI[0], ROI[1], ROI[2], ROI[3]
  #Crop Image
  img_crop = frame[int(y1):int(y1+y2),int(x1):int(x1+x2)]
  cv2.circle(img_crop,(circle[0]-x1, circle[1]-y1),circle[2],(0,255,0),2)
  cv2.imshow("crop", img_crop)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  return ROI,circle

def calc_circle(ROI):
  x1 = ROI[0]
  y1 = ROI[1]
  x2 = ROI[2]
  y2 = ROI[3]
  # 内接円の計算
  x_c,y_c=x1+int(x2/2),y1+int(y2/2)
  circle=[x_c,y_c,int(min(x2,y2)/2)]
  return circle

def calc_ROI(circle):
  x1=circle[0]-circle[2]
  y1=circle[1]-circle[2]
  x2=circle[2]*2
  y2=circle[2]*2
  return [x1,y1,x2,y2]

def pre_image_process(frame,show_state,norm_hist,save_file_path=None):
  '''
  画像の前処理を行う
  frame: 対象とするフレーム
  '''
  #kernel = make_sharp_kernel(0)
  #frame = cv2.cuda.filter2D(frame, -1, kernel).astype("uint8")
  frame_ = cv2.cuda.cvtColor(frame, cv2.COLOR_BGR2GRAY) #画像をグレースケール化する
  if norm_hist:
    frame_=cv2.cuda.equalizeHist(frame_) # ヒストグラム平均化を使う場合
  frame_past=frame_
  frame_past_d= frame_.download()
  if save_file_path != None:
    cv2.imwrite(save_file_path+'pre_image_process.png', frame_past_d)
  return frame_past


def show_progress_bar(total_frame_num, pro_bar_num, current_frame_num, start):
  '''
  処理全体の進捗状況の可視化を行う
  total_frame_num: 対象とする動画の長さ
  pro_bar_num: 進捗を可視化する分割数
  current_frame_num: 現在のフレームの番号
  '''
  pro_bar='['
  percent_progress=int(current_frame_num/(total_frame_num/pro_bar_num))
  for pro in range(percent_progress):
    pro_bar=pro_bar+'#'
  for pro in range(pro_bar_num-percent_progress):
    pro_bar=pro_bar+'.'
  pro_bar=pro_bar+']'+'%d'%(percent_progress*(1/pro_bar_num)*100)+'%'
  elapsed_time = time.time() - start
  # print(pro_bar)
  print("elapsed_time:{0:6.1f}".format(elapsed_time) + "[sec]"+pro_bar)
  return

def match_best(bf, des_past, des_current,threshold=0.2):
  '''
  計算された特徴量点の対応関係の内，最小のものを計算する
  bf: cv2.BFMatcher()
  des_past, des_current:過去と現在の特徴量点のオブジェクト
  threshold: 最大で離れている距離の大きさ
  '''
  matches = bf.match(des_past,des_current) #一つのKPに対して，k個の特徴点を対応付ける(?)
  good,good_train_ind = [],[]
  for m in matches:
      if m.distance<threshold:
          good.append([m])
          good_train_ind.append(m.trainIdx)
  return good,good_train_ind

def match_knn(bf, des_past, des_current, threshold=0.75):
  '''
  計算された特徴量点の対応関係の内，上位2つを計算する
  bf: cv2.BFMatcher()
  des_past, des_current:過去と現在の特徴量点のオブジェクト
  threshold: 二番目に近い特徴とどのくらい離れている場合に採用するかを決める(0~1の間)
  '''
  #matches = bf.knnMatch(des_past, des_current, k=2) #一つのKPに対して，k個の特徴点を対応付ける(?)
  matches = bf.knnMatchAsync(des_past, des_current, k=1) 
  good,good_train_ind = [],[]
  for m,n in matches:
      if m.distance < threshold*n.distance:
          good.append([m])
          good_train_ind.append(m.trainIdx)
  return good,good_train_ind

def visualization_of_correspondence_points(match_type,frame_past,kp_past,frame_current,kp_current,good,save_file_path,_):
  '''
  特徴量点の対応関係を可視化する
  match_type: 対応関係を計算した手法
  frame_past, kp_past, frame_current, kp_current: 対象とする2枚の画像とその画像から計算されたkey point
  good: 実際に対応関係の存在したkey pointのリスト
  save_file_path: 可視化した画像を保存するパス
  '''
  if match_type=='knn':
    img_surf = cv2.drawMatchesKnn(frame_past,kp_past,frame_current,kp_current,good, None,flags=2)
  if match_type=='best':
    #img_surf = cv2.drawMatches(frame_past,kp_past,frame_current,kp_current,good, None,flags=2)
    img_surf = cv2.drawKeypoints(frame_past,kp_past, None,flags=2)
  # 対応関係の可視化
  #os.makedirs(save_file_path, exist_ok=True)
  cv2.imwrite(save_file_path+'img_surf_match_'+'%d'%_+'.png', img_surf)

  #cv2.imshow('color', img_surf)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def affine_mat(df,good,kp_past,kp_current,frame_distance=1):
  '''
  計算されたkey pointの対応から，アフィン変換行列を推定する
  なお，ここでは，estimateAffinePartial2を使用しているので，
  回転、等方性倍率、X並進、Y並進の4つから行列を求める制約付きの推定
  df: 変換行列の保存先
  good: 実際に対応関係の存在したkey pointのリスト
  kp_past,kp_current: 過去と現在の特徴量点
  '''
  kp_past_pts,kp_current_pts = [],[]
  for g in good:
    kp_past_pts.append(kp_past[g[0].queryIdx].pt)
    kp_current_pts.append(kp_current[g[0].trainIdx].pt)
  #TODO ここの関数に渡す順番の確認
  mat,_=cv2.estimateAffinePartial2D(np.array(kp_current_pts),np.array(kp_past_pts))
  #mat,_=cv2.cuda.estimateAffinePartial2D(cp.array(kp_current_pts),cp.array(kp_past_pts))
  #TODO この回転行列から求めた角度が正しいかどうかをテストする
  _df=pd.DataFrame([{'R11':mat[0,0]/frame_distance,'R12':mat[0,1]/frame_distance,'R13':mat[0,2]/frame_distance,
                                  'R21':mat[1,0]/frame_distance,'R22':mat[1,1]/frame_distance,'R23':mat[1,2]/frame_distance,
                                  'dx':mat[0,2]/frame_distance  ,'dy':mat[1,2]/frame_distance  ,'dtheta':np.rad2deg(np.arcsin(mat[1,0]))/frame_distance,
                                  'num of Kp':len(good)}])
  df=pd.concat((df,_df))
  return df

def luminance_check(frame,ROI,df,framenum):
  """
  指定された範囲内の平均輝度を取得する
  """
  x1 = ROI[0]
  y1 = ROI[1]
  x2 = ROI[2]
  y2 = ROI[3]

  frame_crop = frame[int(y1):int(y1+y2),int(x1):int(x1+x2)]

  hsv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
  h,s,v = cv2.split(hsv)
  _df=pd.DataFrame([{'luminance':(np.mean(v))}])
  #__df=pd.DataFrame([{'index':(framenum)}])
  df=pd.concat([df,_df])
  return df  

#sharpness
def make_sharp_kernel(k: int):
  return np.array([
    [-k / 9, -k / 9, -k / 9],
    [-k / 9, 1 + 8 * k / 9, k / 9],
    [-k / 9, -k / 9, -k / 9]
  ], np.float32)

