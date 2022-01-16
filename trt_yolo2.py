# -*- coding:cp949 -*-
import socket
import cv2
import pickle
import struct
import threading
import dlib
import numpy as np
from model import Net
import torch
from imutils import face_utils

import os
import time
import argparse

import cv2

#--------for YOLO--------#
#------------------------#
import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
#------------------------#


#--------Parameter Settings-------#
#---------------------------------#
ip = '114.70.23.41' # ip 주소
port = 5050 # port 번호

# 이미지 사이즈는 전체 이미지 사이즈가 아니라 
# 얼굴 이미지에서 눈으로 인식하여 crop한 2개의 이미지에 대한 사이즈
IMG_SIZE = (34,26)
PATH = './weights/trained.pth'

# dlib의 얼굴을 인식할 수 있는 detector와 predictor를 선언
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')

# 눈을 감았는지 떴는지 판단하는 model을 불러옴
model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()

n_count = 0 

WINDOW_NAME = 'TrtYOLODemo'
#---------------------------------#


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')

    args = parser.parse_args()
    return args

# 이미지에서 눈 부분을 자르는 함수
# eye_points라는 parameter는 37~42, 43~48
def crop_eye(img, eye_points):
    # 37~42, 43~48 이 점을 통해 x1,x2,y1,y2의 좌표를 뽑아냄 
    # opencv에서 (0,0)에 해당하는 점은 왼쪽 맨 위
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    # x의 중점 cx, y의 중점 cy를 구하기
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2  # width
    h = w * IMG_SIZE[1] / IMG_SIZE[0]   # height

    margin_x, margin_y = w / 2, h / 2   # margin

    # margin값을 고려해 x,y좌표를 구해주고 이를 통해 eye_rect 좌표를 생성
    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    # 좌표에 해당하는 이미지를 grayscale로 변경
    eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    # 이 이미지와 위에서 구한 eye_rect 좌표를 return
    return eye_img, eye_rect


# model을 통해 예측
def predict(pred):
  # 들어온 눈 이미지를 transpose하여 model input과 같은 size로 변경
  pred = pred.transpose(1, 3).transpose(2, 3)

  outputs = model(pred)
  # model을 통해 나온 output 즉 예측값을 sigmoid를 통해 확률화시키고 
  # round를 통해 0 또는 1로 변환 (0.5 이하는 0, 초과는 1)
  pred_tag = torch.round(torch.sigmoid(outputs))

  return pred_tag

# 졸음 감지시 관리자에게 사진 전송
def send_img(conn, frame_data):
    size = len(frame_data)
    conn.send(struct.pack(">L", size) + frame_data)
    # print(struct.pack(">L", size) + frame_data)
    print("@@@@@@@send success@@@@@@@")

# 학생에게 물마시라는 경고 출력을 위한 flag 전송
def alarm(conn, text):
    conn.send(text.encode())

def main():

    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)


    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # 소켓 객체를 생성
    s.bind((ip, port)) # 바인드(bind) : 소켓에 주소, 프로토콜, 포트를 할당
    s.listen(2) # 연결 수신 대기 상태(리스닝 수(동시 접속) 설정)
    print('Listening....')

    # 연결 수락(클라이언트 소켓 주소를 반환)
    conn_teacher, addr_teacher = s.accept()
    print("Teacher Connect Success: ", addr_teacher) # 클라이언트 주소 출력

    conn, addr = s.accept()
    print("Student Cam Connect Success: ", addr) # 클라이언트 주소 출력

    data = b"" # 수신한 데이터를 넣을 변수
    payload_size = struct.calcsize(">L")
    
    # 프레임 저장 변수
    n_count = 0
    
    while True:
        # 프레임 수신
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        #print("Frame Size : {}".format(msg_size)) # 프레임 크기 출력


        # 역직렬화(de-serialization) : 직렬화된 파일이나 바이트를 원래의 객체로 복원하는 것
        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes") # 직렬화되어 있는 binary file로 부터 객체로 역직렬화
        img_ori = cv2.imdecode(frame, cv2.IMREAD_COLOR) # 프레임 디코딩


        #-----------깜빡임 인식-----------#
        #---------------------------------#
        
        # cv2.resize로 사이즈를 조절
        img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

        img = img_ori.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # rgb -> gray scale

        # detector를 통해 gray scale image에서 얼굴을 인식
        faces = detector(gray)

        for face in faces:
            # predictor를 통해 face의 좌표에서 landmark를 추정
            shapes = predictor(gray, face)
            shapes = face_utils.shape_to_np(shapes)

            # crop_eye 함수를 통해 눈 부분만 crop하여 
            # 각각 eye_img_l,eye_rect_l,eye_img_r,eye_rect_r에 이미지와 눈 좌표를 저장
            eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
            eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

            # 눈이 인식된 이미지를 model input size에 맞게 조절
            eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
            eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
            eye_img_r = cv2.flip(eye_img_r, flipCode=1)

            # 만약 눈만 인식한 결과를 보고 싶을 경우 아래 주석을 해제
            # cv2.imshow('l', eye_img_l)
            # cv2.imshow('r', eye_img_r)

            eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
            eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)

            # numpy 배열인 이미지를 torch.from_numpy를 통해 tensor로 변환
            eye_input_l = torch.from_numpy(eye_input_l)
            eye_input_r = torch.from_numpy(eye_input_r)

            # model에 해당하는 predict을 통해 눈을 감았는지 판단
            pred_l = predict(eye_input_l)
            pred_r = predict(eye_input_r)

            # 두 눈을 다 감고있다면 n_count + 1
            if pred_l.item() == 0.0 and pred_r.item() == 0.0:
                n_count+=1

            # 아니라면 n_count 초기화
            else:
                n_count = 0

            print("count: ", n_count, "/ 50", end='\t\t')

            # 100 이상이면 관리자에게는 사진전송, 학생에게는 물마시라는 flag 전송
            if n_count > 50 :
                cv2.putText(img, "Wake up", (120,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                thread = threading.Thread(target = send_img, args = (conn_teacher, frame_data))
                thread.start()
                print("Send Wake up Message")

                text = '1'  # 물마시라는 flag
                thread = threading.Thread(target = alarm, args = (conn, text))
                thread.start()

                n_count = 0

        #------물체 인식(물 마시는지)-------#
        #---------------------------------#

        # 영상 출력
        # cv2.imshow('video',frame)

        img2 = img_ori.copy()
        # 이미지에서 물체인식 진행
        boxes, confs, clss = trt_yolo.detect(img2, 0.3)
        # 감지된 곳에 bbox
        img2 = vis.draw_bboxes(img2, boxes, confs, clss)
        # 감지된 게 있다면
        if len(clss):
            # 감지된 객체 cls를 넘버링
            for i, cls in enumerate(clss):
                cls_name = cls_dict.get(int(cls))
                print(cls_name,"\t", confs[i])

                # cup이나 bottle이 감지된다면 알림 종료 flag 전송
                if cls_name == 'cup' or cls_name == 'bottle':
                    text = '0'  # 알림 종료 flag
                    thread = threading.Thread(target = alarm, args = (conn, text))
                    thread.start()

        # 1초 마다 키 입력 상태를 받음
        if cv2.waitKey(1) == ord('q') : # q를 입력하면 종료
            print("close")
            s.close()
            break

        
if __name__ == "__main__":
    main()