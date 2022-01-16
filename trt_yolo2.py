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
ip = '114.70.23.41' # ip �ּ�
port = 5050 # port ��ȣ

# �̹��� ������� ��ü �̹��� ����� �ƴ϶� 
# �� �̹������� ������ �ν��Ͽ� crop�� 2���� �̹����� ���� ������
IMG_SIZE = (34,26)
PATH = './weights/trained.pth'

# dlib�� ���� �ν��� �� �ִ� detector�� predictor�� ����
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')

# ���� ���Ҵ��� ������ �Ǵ��ϴ� model�� �ҷ���
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

# �̹������� �� �κ��� �ڸ��� �Լ�
# eye_points��� parameter�� 37~42, 43~48
def crop_eye(img, eye_points):
    # 37~42, 43~48 �� ���� ���� x1,x2,y1,y2�� ��ǥ�� �̾Ƴ� 
    # opencv���� (0,0)�� �ش��ϴ� ���� ���� �� ��
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    # x�� ���� cx, y�� ���� cy�� ���ϱ�
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2  # width
    h = w * IMG_SIZE[1] / IMG_SIZE[0]   # height

    margin_x, margin_y = w / 2, h / 2   # margin

    # margin���� ����� x,y��ǥ�� �����ְ� �̸� ���� eye_rect ��ǥ�� ����
    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    # ��ǥ�� �ش��ϴ� �̹����� grayscale�� ����
    eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    # �� �̹����� ������ ���� eye_rect ��ǥ�� return
    return eye_img, eye_rect


# model�� ���� ����
def predict(pred):
  # ���� �� �̹����� transpose�Ͽ� model input�� ���� size�� ����
  pred = pred.transpose(1, 3).transpose(2, 3)

  outputs = model(pred)
  # model�� ���� ���� output �� �������� sigmoid�� ���� Ȯ��ȭ��Ű�� 
  # round�� ���� 0 �Ǵ� 1�� ��ȯ (0.5 ���ϴ� 0, �ʰ��� 1)
  pred_tag = torch.round(torch.sigmoid(outputs))

  return pred_tag

# ���� ������ �����ڿ��� ���� ����
def send_img(conn, frame_data):
    size = len(frame_data)
    conn.send(struct.pack(">L", size) + frame_data)
    # print(struct.pack(">L", size) + frame_data)
    print("@@@@@@@send success@@@@@@@")

# �л����� �����ö�� ��� ����� ���� flag ����
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


    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # ���� ��ü�� ����
    s.bind((ip, port)) # ���ε�(bind) : ���Ͽ� �ּ�, ��������, ��Ʈ�� �Ҵ�
    s.listen(2) # ���� ���� ��� ����(������ ��(���� ����) ����)
    print('Listening....')

    # ���� ����(Ŭ���̾�Ʈ ���� �ּҸ� ��ȯ)
    conn_teacher, addr_teacher = s.accept()
    print("Teacher Connect Success: ", addr_teacher) # Ŭ���̾�Ʈ �ּ� ���

    conn, addr = s.accept()
    print("Student Cam Connect Success: ", addr) # Ŭ���̾�Ʈ �ּ� ���

    data = b"" # ������ �����͸� ���� ����
    payload_size = struct.calcsize(">L")
    
    # ������ ���� ����
    n_count = 0
    
    while True:
        # ������ ����
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        #print("Frame Size : {}".format(msg_size)) # ������ ũ�� ���


        # ������ȭ(de-serialization) : ����ȭ�� �����̳� ����Ʈ�� ������ ��ü�� �����ϴ� ��
        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes") # ����ȭ�Ǿ� �ִ� binary file�� ���� ��ü�� ������ȭ
        img_ori = cv2.imdecode(frame, cv2.IMREAD_COLOR) # ������ ���ڵ�


        #-----------������ �ν�-----------#
        #---------------------------------#
        
        # cv2.resize�� ����� ����
        img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

        img = img_ori.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # rgb -> gray scale

        # detector�� ���� gray scale image���� ���� �ν�
        faces = detector(gray)

        for face in faces:
            # predictor�� ���� face�� ��ǥ���� landmark�� ����
            shapes = predictor(gray, face)
            shapes = face_utils.shape_to_np(shapes)

            # crop_eye �Լ��� ���� �� �κи� crop�Ͽ� 
            # ���� eye_img_l,eye_rect_l,eye_img_r,eye_rect_r�� �̹����� �� ��ǥ�� ����
            eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
            eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

            # ���� �νĵ� �̹����� model input size�� �°� ����
            eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
            eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
            eye_img_r = cv2.flip(eye_img_r, flipCode=1)

            # ���� ���� �ν��� ����� ���� ���� ��� �Ʒ� �ּ��� ����
            # cv2.imshow('l', eye_img_l)
            # cv2.imshow('r', eye_img_r)

            eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
            eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)

            # numpy �迭�� �̹����� torch.from_numpy�� ���� tensor�� ��ȯ
            eye_input_l = torch.from_numpy(eye_input_l)
            eye_input_r = torch.from_numpy(eye_input_r)

            # model�� �ش��ϴ� predict�� ���� ���� ���Ҵ��� �Ǵ�
            pred_l = predict(eye_input_l)
            pred_r = predict(eye_input_r)

            # �� ���� �� �����ִٸ� n_count + 1
            if pred_l.item() == 0.0 and pred_r.item() == 0.0:
                n_count+=1

            # �ƴ϶�� n_count �ʱ�ȭ
            else:
                n_count = 0

            print("count: ", n_count, "/ 50", end='\t\t')

            # 100 �̻��̸� �����ڿ��Դ� ��������, �л����Դ� �����ö�� flag ����
            if n_count > 50 :
                cv2.putText(img, "Wake up", (120,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                thread = threading.Thread(target = send_img, args = (conn_teacher, frame_data))
                thread.start()
                print("Send Wake up Message")

                text = '1'  # �����ö�� flag
                thread = threading.Thread(target = alarm, args = (conn, text))
                thread.start()

                n_count = 0

        #------��ü �ν�(�� ���ô���)-------#
        #---------------------------------#

        # ���� ���
        # cv2.imshow('video',frame)

        img2 = img_ori.copy()
        # �̹������� ��ü�ν� ����
        boxes, confs, clss = trt_yolo.detect(img2, 0.3)
        # ������ ���� bbox
        img2 = vis.draw_bboxes(img2, boxes, confs, clss)
        # ������ �� �ִٸ�
        if len(clss):
            # ������ ��ü cls�� �ѹ���
            for i, cls in enumerate(clss):
                cls_name = cls_dict.get(int(cls))
                print(cls_name,"\t", confs[i])

                # cup�̳� bottle�� �����ȴٸ� �˸� ���� flag ����
                if cls_name == 'cup' or cls_name == 'bottle':
                    text = '0'  # �˸� ���� flag
                    thread = threading.Thread(target = alarm, args = (conn, text))
                    thread.start()

        # 1�� ���� Ű �Է� ���¸� ����
        if cv2.waitKey(1) == ord('q') : # q�� �Է��ϸ� ����
            print("close")
            s.close()
            break

        
if __name__ == "__main__":
    main()