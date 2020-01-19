# 모듈 선언
import os
import numpy as np
import tensorflow as tf
import argparse
import sys
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import RPi.GPIO as GPIO
import pygame
import board
import neopixel
import serial

# GPIO Pin 선언 및 사운드파일 경로설정
pygame.mixer.init()
Bell = pygame.mixer.Sound("/home/pi/025832392-magic-idea-05.wav")
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
Photo_SENSOR = 16
LED_animal_relay = 5
Motion_CENTER = 13
Motion_LEFT = 19
Motion_RIGHT = 26
RED = 14
GREEN = 15

#GPIO input, output 선언
GPIO.setup(RED, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(GREEN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(Photo_SENSOR,GPIO.IN)
GPIO.setup(Motion_CENTER,GPIO.IN)
GPIO.setup(Motion_LEFT,GPIO.IN)
GPIO.setup(Motion_RIGHT,GPIO.IN)
GPIO.setup(LED_animal_relay, GPIO.OUT, initial=GPIO.LOW)

# 카메라 해상도 설정
cnt = 0   
IM_WIDTH = 1280
IM_HEIGHT = 720

# 현재 디렉토리를 모듈 경로로 추가.
sys.path.append('..')

# 원할한 interrupt 처리를 위한 변수선언
Interruptcheck=1
Pass=0

# 물체인식을 위한 유틸리티 import
from utils import label_map_util
from utils import visualization_utils as vis_util

# 물체 인식을 위해 사용할 사전 학습된 모델 불러오기
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# 현재 디렉토리를 CWD_PATH로 지정
CWD_PATH = os.getcwd()

# 학습완료된 그래프 pb파일 경로를 지정
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# 레이블(분류할 물체 : ex : 사람, 고양이)들이 저장된 pbtxt 파일의 경로 지정
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# 현재 사용하는 pbtxt 파일의 레이블 개수
NUM_CLASSES = 90

# 레이블별로 id가 할당된 레이블맵 불러오기
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# 그래프로 저장된 pb파일을 읽어서 그래프를 생성(텐서플로우에 로드)시킴
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# input, output tensor을 구성하는 변수를 정의함

# Input tensor는 이미지(사진) 한가지 변수로 구성됨.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensor는 탐지 박스, 점수, 종류 총 세가지 변수로 구성됨
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# 추가로 감지된 물체의 수를 나타내는 변수를 정의
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# 사진 촬영을 위한 카메라 옵션 설정.
camera = PiCamera()
camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
rawCapture.truncate(0)

# 동물 판별 및 퇴치함수 Action 정의
def Action():

    # 전역변수 Interruptcheck의 값을 0으로 변환
    # 이는 while loop문에서 실행하는 조도센서 제어와 
    # 각 모션센서의 원할한 Interrupt 처리를 위해 필요
    global Interruptcheck
    Interruptcheck = 0
    
    # relay(LED투광기)의 초기상태를 off로 설정
    GPIO.output(LED_animal_relay, GPIO.LOW)
    
    # 조도센서의 상태를 알기 위한 지역변수 Photo_Input 선언
    Photo_Input = GPIO.input(Photo_SENSOR)
    LED_status = "text"
    
    # while문의 반복횟수를 제어하기 위한 지역변수 Count 선언    
    Count = 1
    
    # Count가 4 미만의 값을 가질때 수행되는 while문 선언
    while Count < 4:
    
        # 사진을 촬영하고 촬영된 이미지파일을 행렬연산이 가능한 Input Tensor로 변환
        camera.capture(rawCapture, format="bgr",use_video_port=True)
        frame1 = rawCapture.array
        frame = np.copy(frame1)
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Input Tensor를 텐서플로우 모델에 대입하여 Output Tensor를 얻어냄
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        
        # 현재의 Output Tensor는 단순한 결과값이므로 이를 변수화하는 알고리즘이 추가로 필요
        
        # Output Tensor 중 np.squeeze(scores)(탐지된 물체의 일치도를 나타내는 행렬) 행렬에서 
        # 가장 높은 정확도를 가지는 항의 값을 나타내는 지역변수 probabilities 선언                       
        probabilities = max(np.squeeze(scores))
        
        # np.squeeze(scores)에서 가장 높은 정확도를 가지는 행렬의 주소를 나타내는 
        # 변수 pro_index 선언 및 터미널 상에 표시
        pro_index = np.where(np.squeeze(scores) == probabilities)
        print('score :', probabilities )
               
        # pro_index의 값(probabilities의 행렬주소)를 물체의 label을 나타내는 행렬에 
        # 대입하여 가장 높은 정확도를 가진 물체의 label을 나타내는 변수 labels 선언 및 터미널 상에 표시        
        labels = np.squeeze(classes)[pro_index].astype(np.int32)
        print('labels :', labels )
        
        # 조도센서의 상태에 따라 가로등의 ON,OFF 상태를 제어하는 조건문
        
        # 조도센서의 Input이 1일 때(어두운 상태)
        if Photo_Input == 1:
            
            # 지역변수 LED_status의 값을 "on"으로 변환하고 
            # neopixel LED의 빛을 white로 변환하는 제어코드 실행
            LED_status = "on"
            os.system("sudo python3 White_On.py")
        
        # 조도센서의 Input이 0일 때(밝은 상태)
        else:
            # 지역변수 LED_status의 값을 "off"로 변환하고 
            # neopixel LED의 빛을 off하는 제어코드 실행
            LED_status = "off"
            os.system("sudo python3 Light_Off.py")
        
        # 위에서 변수화된 물체의 정확도가 만족되고 주변이 어두운 상태일때 실행
        if probabilities>0.3 and LED_status == "on":
            
            # 변수화된 물체가 사람으로 판별되었을 경우
            if labels == 1:
                
                print('Person Detected!!(night)')
                
                # while문의 반복횟수를 결정하는 지역변수 Count를 1로 초기화
                Count = 1
                
                # neopixel LED의 빛을 red로 변환하는 제어코드 실행
                os.system("sudo python3 Red_On.py")

                # 차랑용 단말기 작동을 위한 RF 신호 송신코드 실행
                os.system("python3 commu.py")
                
                # relay(LED투광기) on 
                GPIO.output(LED_animal_relay, GPIO.HIGH)
                
                # 1.5초 간격으로 미리 설정된 사운드파일 실행
                for i in range(4):
                    Bell.play()
                    time.sleep(1.5)
                    
                # relay(LED투광기) off
                GPIO.output(LED_animal_relay, GPIO.LOW)
                
                # neopixel LED의 빛을 다시 white로 변환하는 제어코드 실행
                os.system("sudo python3 White_On.py")
                
            # 변수화된 물체가 동물로 판별되었을 경우(개, 고양이 등등)
            elif labels == 16 or labels == 17 or labels == 18 or labels == 19 or labels == 20 or labels == 21:
                
                print('Animal Detected!!(night)')
                                
                # while문의 반복횟수를 결정하는 지역변수 Count를 1로 초기화
                Count = 1
                
                # neopixel LED의 빛을 red로 변환하는 제어코드 실행
                os.system("sudo python3 Red_On.py") 
                
                # 차랑용 단말기 작동을 위한 RF 신호 송신코드 실행
                os.system("python3 commu.py")
                
                # relay(LED투광기) on 
                GPIO.output(LED_animal_relay, GPIO.HIGH)
                               
                # 1.5초 간격으로 미리 설정된 사운드파일 실행
                for i in range(4):
                    Bell.play()
                    time.sleep(1.5)
                
                # relay(LED투광기) off    
                GPIO.output(LED_animal_relay, GPIO.LOW)
                
                # neopixel LED의 빛을 다시 white로 변환하는 제어코드 실행
                os.system("sudo python3 White_On.py")
            
            # 판별결과 동물이나 사람이 감지되지 않았을 경우 지역변수 Count의 값을 1 추가
            else:
                Count += 1
                print('Detection Failed')
        
        # 위에서 변수화된 물체의 정확도가 만족되고 주변이 밝은 상태일때 실행    
        elif probabilities>0.3 and LED_status == "off":
            
            # 변수화된 물체가 사람으로 판별되었을 경우
            if labels == 1:
                
                print('Person Detected!!(day)')
                
                # while문의 반복횟수를 결정하는 지역변수 Count를 1로 초기화
                Count = 1

                # 차랑용 단말기 작동을 위한 RF 신호 송신코드 실행                
                os.system("python3 commu.py")
                
                # relay(LED투광기) on
                GPIO.output(LED_animal_relay, GPIO.HIGH)
                
                # 1.5초 간격으로 미리 설정된 사운드파일 실행
                for i in range(4):
                    Bell.play()
                    time.sleep(1.5)
                    
                # relay(LED투광기) off    
                GPIO.output(LED_animal_relay, GPIO.LOW)
            
            # 변수화된 물체가 동물로 판별되었을 경우(개, 고양이 등등)        
            elif labels == 16 or labels == 17 or labels == 18 or labels == 19 or labels == 20 or labels == 21:
                
                print('Animal Detected!!(day)')
                
                # while문의 반복횟수를 결정하는 지역변수 Count를 1로 초기화
                Count = 1

                # 차랑용 단말기 작동을 위한 RF 신호 송신코드 실행                
                os.system("python3 commu.py")
                
                # relay(LED투광기) on
                GPIO.output(LED_animal_relay, GPIO.HIGH)
                
                # 1.5초 간격으로 미리 설정된 사운드파일 실행
                for i in range(4):
                    Bell.play()
                    time.sleep(1.5)
                    
                # relay(LED투광기) off    
                GPIO.output(LED_animal_relay, GPIO.LOW)
            
            # 판별결과 동물이나 사람이 감지되지 않았을 경우 지역변수 Count의 값을 1 추가    
            else:
                Count += 1
                print('Detection Failed')
        
        # 판별결과 동물이나 사람이 감지되지 않았을 경우 지역변수 Count의 값을 1 추가            
        else:
            Count += 1
            print('Detection Failed')
        
        rawCapture.truncate(0)
        
    # while loop문 탈출 시 return    
    return 0

# 왼쪽에 위치한 모션센서에서 모션이 감지되었을 때의 실행함수 Process_left 정의          
def Process_left(gpio):
        
        # 전역변수 Interruptcheck의 값이 1인 경우 함수 실행
        if Interruptcheck == 1:
            
            # 현재 모듈의 상태를 나타내는 3색 LED의 색을 RED로 변환(감지상태) 
            GPIO.output(GREEN, GPIO.LOW)
            GPIO.output(RED, GPIO.HIGH)
            print("Left start")
            
            # 서보모터를 왼쪽으로 회전시키는 servoblaster 명령어 실행
            os.system("echo 2=83% > /dev/servoblaster")
            # 모터 회전시간 확보
            time.sleep(0.5)        
            
            # 동물 감지 및 퇴치함수 Action 실행
            Action()
            
            # 서보모터를 다시 중앙으로 위치시키는 servoblaster 명령어 실행
            os.system("echo 2=43% > /dev/servoblaster")
            # 모터 회전시간 확보
            time.sleep(0.5)        
            
            # 현재 모듈의 상태를 나타내는 3색 LED의 색을 GREEN으로 변환(대기상태) 
            GPIO.output(RED, GPIO.LOW)   
            GPIO.output(GREEN, GPIO.HIGH)
            print("Left end")        
            
        else:
            
            # 전역변수 Interruptcheck의 값이 1일때는(누적된 Interrupt)
            # 의도되지 않은 Interrupt이므로 아무런 동작없이 전역변수 Pass의
            # 값만 1로 변환 후 return
            print("pass1")
            global Pass
            Pass = 1
            return 0

# 중앙부의 위치한 모션센서에서 모션이 감지되었을 때의 실행함수 Process_center 정의        
def Process_center(gpio):
    
        # 전역변수 Interruptcheck의 값이 1인 경우 함수 실행
        if Interruptcheck == 1:
            
            # 현재 모듈의 상태를 나타내는 3색 LED의 색을 RED로 변환(감지상태) 
            GPIO.output(GREEN, GPIO.LOW)
            GPIO.output(RED, GPIO.HIGH)
            print("Center Start")
            
            # 동물 감지 및 퇴치함수 Action 실행
            Action()
            
            # 현재 모듈의 상태를 나타내는 3색 LED의 색을 GREEN으로 변환(대기상태)
            GPIO.output(RED, GPIO.LOW)   
            GPIO.output(GREEN, GPIO.HIGH)
            print("Center End")

        else:            
            
            # Interruptcheck의 값이 1일 경우 주석 330~331과 같은 동작      
            print("pass2")            
            global Pass
            Pass = 1
            return 0

# 오른쪽에 위치한 모션센서에서 모션이 감지되었을 때의 실행함수 Process_right 정의              
def Process_right(gpio):
        
        # 전역변수 Interruptcheck의 값이 1인 경우 함수 실행
        if Interruptcheck == 1:    
            
            # 현재 모듈의 상태를 나타내는 3색 LED의 색을 RED로 변환(감지상태) 
            GPIO.output(GREEN, GPIO.LOW)
            GPIO.output(RED, GPIO.HIGH)
            print("Right start")            
            
            # 서보모터를 오른쪽으로 회전시키는 servoblaster 명령어 실행
            os.system("echo 2=3% > /dev/servoblaster")
            # 모터 회전시간 확보
            time.sleep(0.5)
            
            Action()

            # 서보모터를 다시 중앙으로 위치시키는 servoblaster 명령어 실행
            os.system("echo 2=43% > /dev/servoblaster")            
            # 모터 회전시간 확보
            time.sleep(0.5)        
            
            # 현재 모듈의 상태를 나타내는 3색 LED의 색을 GREEN으로 변환(대기상태)
            GPIO.output(RED, GPIO.LOW)        
            GPIO.output(GREEN, GPIO.HIGH)
            print("Right End")
        
        else:
        
            # Interruptcheck의 값이 1일 경우 주석 330~331과 같은 동작 
            print("pass3")
            global Pass
            Pass = 1
            return 0        

# 각 모션센서에서 Interrupt가 발생하였을 경우 지정된 callback 함수를 실행    
GPIO.add_event_detect(Motion_CENTER,GPIO.FALLING,Process_center, 2000)
GPIO.add_event_detect(Motion_LEFT,GPIO.FALLING,Process_left,2000)
GPIO.add_event_detect(Motion_RIGHT,GPIO.FALLING,Process_right,2000)

# 3개의 모션센서가 Interrupt를 대기하는 중에 실행되는 while loop문
try:
    while True:
        
        # Python3의 Interrupt 처리 명령어는 3개의 센서가 각각 별개로 Interrupt를
        # 처리하게 하므로 의도되지 않은 Interrupt가 발상하는데 이를 위해 
        # 별도의 Software적인 처리가 필요함
        
        # 전역변수 Interruptcheck는 하나의 센서가 Interrupt를 처리할 때 다른 센서가
        # Interrupt를 처리하지 못하도록 하는 역할을 수행함
        global Interruptcheck
        
        # 대기상태에서의 가로등 조명 on,off 제어 
        if Interruptcheck == 1:
            
            if GPIO.input(Photo_SENSOR) == 1:
                os.system("sudo python3 White_On.py")
            else:
                os.system("sudo python3 Light_Off.py")
                
        time.sleep(0.1)
        
        # 전역변수 Pass는 의도되지 않은 Interrupt를 처리한 후 다시 
        # 정상적으로 Interrupt를 처리하는 대기상태로 전환시켜주는 역할을 함
        
        # 각 process함수에서 return과 동시에 전역변수 Pass의 값이 1이
        # 되면 실행되어 전역변수 Interruptcheck, Pass의 값을 초기값으로 변환
        if Pass == 1:
            Interruptcheck=1            
            global Pass
            Pass = 0
                        
except KeyboardInterrupt:
    is_running = False
    GPIO.cleanup()
