# -*- coding:cp949 -*-

import cv2
import socket
import struct
import pickle
import threading
import time

# �����ö�� ��� ���
def alarm(client_socket):
    flag = 0
    while True:
        water = client_socket.recv(100).decode()    # nx board���Լ� ������ �����ߴ��� �޾ƿ�
        if int(water) == 1: # ���������� �ȵƴٸ�
            print("���� ���ü���")  # ���� ���ö�� ���
            flag = 1
        elif int(water) == 0:   # ���������� �ƴٸ�
            if flag == 1:
                print("�˸��� �����մϴ�")  # ��� ����
                flag = 0
            
        
ip = '114.70.23.41' # ip �ּ�
port = 5050 # port ��ȣ

# ���� ��ü�� ���� �� ����
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((ip, port))
print('���� ����')

# ī�޶� ����
camera = cv2.VideoCapture(0)

# ũ�� ����
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640); # ����
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); # ����

# ���ڵ� �Ķ����
# jpg�� ��� cv2.IMWRITE_JPEG_QUALITY�� �̿��Ͽ� �̹����� ǰ���� ����
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

# ��� ����� ���� thread
thread = threading.Thread(target = alarm, args = (client_socket, ))
thread.start()

while True:
    ret, frame = camera.read() # ī�޶� ������ �б�
    frame = cv2.flip(frame, 1)
    cv2.imshow('check', frame)
    result, frame = cv2.imencode('.jpg', frame, encode_param) # ������ ���ڵ�
    # ����ȭ(serialization) : ȿ�������� �����ϰų� ��Ʈ������ ������ �� ��ü�� �����͸� �ٷ� ���� �����ϴ� ��
    # binary file : ��ǻ�� ����� ó�� ������ ���� ���� �������� ���ڵ��� �����͸� ����
    data = pickle.dumps(frame, 0) # �������� ����ȭȭ�Ͽ� binary file�� ��ȯ
    size = len(data)
    # print("Frame Size : ", size) # ������ ũ�� ���

    
    # ������(������) ����
    client_socket.sendall(struct.pack(">L", size) + data)
    # print(struct.pack(">L", size) + data)

    if cv2.waitKey(1) == ord('q') : # q�� �Է��ϸ� ����
            cv2.destroyAllWindows()
            break
    
    
# �޸𸮸� ����
camera.release()