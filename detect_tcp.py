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

#from torch2trt import torch2trt

ip = '114.70.23.41' # ip 주소
port = 5050 # port 번호

IMG_SIZE = (34,26)
PATH = './weights/trained.pth'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')

model = Net()
model.load_state_dict(torch.load(PATH))
model.eval()

n_count = 0 

def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


def predict(pred):
    pred = pred.transpose(1, 3).transpose(2, 3)

    outputs = model(pred)

    pred_tag = torch.round(torch.sigmoid(outputs))

    return pred_tag


def send_img(conn, frame_data):
    size = len(frame_data)
    conn.send(struct.pack(">L", size) + frame_data)
    # print(struct.pack(">L", size) + frame_data)
    print("@@@@@@@send success@@@@@@@")


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
    print("Frame Size : {}".format(msg_size)) # 프레임 크기 출력


    # 역직렬화(de-serialization) : 직렬화된 파일이나 바이트를 원래의 객체로 복원하는 것
    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes") # 직렬화되어 있는 binary file로 부터 객체로 역직렬화
    img_ori = cv2.imdecode(frame, cv2.IMREAD_COLOR) # 프레임 디코딩


    #-----------Add by Chan-----------#
    #---------------------------------#
    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])


        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        # cv2.imshow('l', eye_img_l)
        # cv2.imshow('r', eye_img_r)

        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)


        eye_input_l = torch.from_numpy(eye_input_l)
        eye_input_r = torch.from_numpy(eye_input_r)


        pred_l = predict(eye_input_l)
        pred_r = predict(eye_input_r)

        if pred_l.item() == 0.0 and pred_r.item() == 0.0:
            n_count+=1

        else:
            n_count = 0


        if n_count > 50:
            cv2.putText(img, "Wake up", (120,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            print("Send Wake up Message")

    #---------------------------------#
        print(n_count)
        if n_count > 50 :
            thread = threading.Thread(target = send_img, args = (conn_teacher, frame_data))
            thread.start()
            n_count = 0


    # 영상 출력
    # cv2.imshow('video',frame)
    
    # 1초 마다 키 입력 상태를 받음
    if cv2.waitKey(1) == ord('q') : # q를 입력하면 종료
        print("close")
        s.close()
        break

