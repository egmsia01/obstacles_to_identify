# -------------------------------------#
#   调用摄像头或者视频进行检测
#   调用摄像头直接运行即可
#   调用视频可以将cv2.VideoCapture()指定路径
# -------------------------------------#
from distutils.fancy_getopt import fancy_getopt
from math import fabs
from random import randint
import threading
import time
from unittest import case
import cv2
from matplotlib.pyplot import flag
import numpy as np
import tensorflow as tf
from PIL import Image
from yolo import YOLO
import socket

def sockText():
    frist = True
    flag = True
    socketNUmber = 0
    coordinate_x = scaled_f_up.size[0]/3
    coordinate_y = scaled_f_up.size[1]/3

    try:
        print('宽：%d,高：%d'%(scaled_f_up.size[0],scaled_f_up.size[1]))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket创建成功!")
        sock.bind(('192.168.1.218', 8080))
        print('端口绑定成功!')
        # 最多监听5个设备连接
        sock.listen(5)
    except:
        print("加载socket失败!")

    print("等待客户端连接...")
    conn, addr = sock.accept()
    print("连接成功!")
    print(addr)
    conn.settimeout(30)

    while True:
        if frist:
            conn.send(b"isConnect\n")
            frist = False
        while socketNUmber < number:
            if 0 < numx[number - 1] < 213:
                conn.send(b"leftfront\n")
                time.sleep(2)
                
            elif numx[number - 1] > 427:
                conn.send(b"rightfront\n")
                time.sleep(2)
                
            elif 214 < numx[number - 1] < 526:
                conn.send(b"front\n")
                time.sleep(2)
                
            
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 初始化消息队列数组
numx = []
numy = []
mqstr = []
number = 0
#加载YOLO视觉模块
yolo = YOLO()

# -------------------------------------#
#   调用摄像头
# -------------------------------------#
capture = cv2.VideoCapture(0)
# capture = cv2.VideoCapture(2)

fps = 0.0

# 初始化程序计时器
time_start = time.time()

# -------------------------------------#
# 物体实际距离：actual_distance   5
# 物体实际高度：actual_high       (高度) car:1.5  people:1.7  bike:1.2
# 物体像素高度：pixel_high        (高度) car:291  people:283 bike:212
# 焦距focus =（pixel_high * actual_distance）/ actual_high
# 焦距focus已知后，就可以根据像素高度测出物体实际距离，公式如下：actual_distance =（actual_high * focus）/pixel_high
# -------------------------------------#
# car：focus_car = (291 * 5) / 1.5 = 970
# people：focus_people = (283 * 5) / 1.7 = 714
# bike：focus_bike = (212 * 5) / 1.2 = 514
# 实际距离计算
# actual_distance_car = (1.5 * 970) / 291 = 5
# actual_distance_people = (1.7 * 714) / 283 = 5
# actual_distance_bike = (1.2 * 514) / 212 = 5
# -------------------------------------#

while True:
    
    # 计时开始
    t1 = time.time()
    # 读取某一帧
    for i in range(2):
        ref, frame = capture.read()
        # Scaling Up the image 1.2 times by specifying both scaling factors
    # scale_up_x = 1.4
    # scale_up_y = 1.1
    scale_up_x = 1
    scale_up_y = 1
    # Scaling Down the image 0.6 times specifying a single scale factor.
    scaled_f_up = cv2.resize(frame, None, fx = scale_up_x, fy = scale_up_y, interpolation = cv2.INTER_LINEAR)
    # 格式转变，BGRtoRGB
    scaled_f_up = cv2.cvtColor(scaled_f_up,cv2.COLOR_BGR2RGB)
    # 转变成Image
    scaled_f_up = Image.fromarray(np.uint8(scaled_f_up))

    if flag:
        # 启动Socket线程
        thsoc = threading.Thread(target = sockText)
        thsoc.start()
        flag = False
    # 进行检测
    scaled_f_up, predicted_classes, x, y = yolo.detect_image(scaled_f_up)  # 数据在后三个变量中
    # 打印x,y
    print("x: %.2f" % int(x), "y: %.2f" % y)
    # 计时结束3.
    time_end = time.time()
    # 获取图像的尺寸
    print('宽：%d,高：%d'%(scaled_f_up.size[0],scaled_f_up.size[1]))
    
    # 进入消息队列条件
    if (time_end - time_start) > 1:
        numx.append(x)
        numy.append(y)
        # mqstr.append(label)
        number = number + 1 

    scaled_f_up = np.array(scaled_f_up)
    # RGBtoBGR满足opencv显示格式
    scaled_f_up = cv2.cvtColor(scaled_f_up, cv2.COLOR_RGB2BGR)
    fps = (fps + (1. / (time.time() - t1))) / 2
    print("fps = %.2f" % fps)
    scaled_f_up = cv2.putText(scaled_f_up, "fps = %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("video", scaled_f_up)

    c = cv2.waitKey(30) & 0xff
    if c == 27:
        capture.release()
        break
