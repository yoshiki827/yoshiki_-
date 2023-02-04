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


def doukousuitei(video,save_file_path,startframe,time):



    df = pd.DataFrame()

    os.makedirs(save_file_path, exist_ok=True)
    save_file_path = save_file_path+'/'

    video.set(cv2.CAP_PROP_POS_FRAMES,startframe)#動画のスタート位置
    total_frame_num =time#video.get(cv2.CAP_PROP_FRAME_COUNT) #- 30*1
    total_cx=0
    total_cy=0
    total_h=0
    total_w=0
    
    v_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    v_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(v_width)
    ret, frame = video.read()

    ROI = cv2.selectROI('Select ROIs', frame, fromCenter = False, showCrosshair = False)
    cv2.rectangle(frame,
              pt1=(50, 150),
              pt2=(125, 250),
              color=(0, 255, 0),
              thickness=3,
              lineType=cv2.LINE_4,
              shift=0)
    for _ in range(int(total_frame_num)):
        ret, frame = video.read()
        
        cv2.rectangle(frame,pt1=(ROI[0],0),pt2=(int(v_width), ROI[1]),color=(255, 255, 255),thickness=-1,lineType=cv2.LINE_4,shift=0)
        cv2.rectangle(frame,pt1=(ROI[0]+ROI[2],ROI[1]),pt2=(int(v_width), int(v_height)),color=(255, 255, 255),thickness=-1,lineType=cv2.LINE_4,shift=0)    
        cv2.rectangle(frame,pt1=(0,ROI[1]+ROI[3]),pt2=(ROI[0]+ROI[2], int(v_height)),color=(255, 255, 255),thickness=-1,lineType=cv2.LINE_4,shift=0)
        cv2.rectangle(frame,pt1=(0,0),pt2=(ROI[0], ROI[1]+ROI[3]),color=(255, 255, 255),thickness=-1,lineType=cv2.LINE_4,shift=0)

        frame_ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)




        ret,thresh = cv2.threshold(frame_,20,255,cv2.THRESH_BINARY) 
        #cv2.imwrite(save_file_path+'pre_image_process_'+'%d'%_+'.png', thresh)
        
        thresh = cv2.bitwise_not(thresh)
        bimg = thresh // 4 + 255 * 3 //4
        resimg = cv2.merge((bimg,bimg,bimg)) 
        cv2.imwrite(save_file_path+'pre_image_process_'+'%d'%_+'.png', resimg)
        contours,hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        len_cnt_max=0
        for i, cnt in enumerate(contours):
            # 楕円フィッティング
            
            if len(cnt) >=len_cnt_max:
                
                len_cnt_max=len(cnt)
                cnt_max=cnt
                #print(ellipse)

                
        
        ellipse = cv2.fitEllipse(cnt_max)
        
        cx = int(ellipse[0][0])
        cy = int(ellipse[0][1])
        h = int(ellipse[1][0])
        w = int(ellipse[1][1])

        #print(h,w)
        
        total_cx=total_cx+cx
        total_cy=total_cy+cy
        total_h=total_h+h
        total_w=total_w+w
        # 楕円描画
        resimg = cv2.ellipse(resimg,ellipse,(255,0,0),2)
        cv2.drawMarker(resimg, (cx,cy), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        #cv2.putText(resimg, str(i+1), (cx+3,cy+3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,80,255), 1,cv2.LINE_AA)

        #cv2.imshow('resimg',resimg)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    average_cx=total_cx/10
    average_cy=total_cy/10
    average_h=total_h/10
    average_w=total_w/10

    return average_cx,average_cy,average_h,average_w
    #print(average_cx,average_cy)

