# -*- coding:cp949 -*-

import cv2
import socket
import struct
import pickle
import threading
import time

def recv_img(client_socket):
    data = b""
    payload_size = struct.calcsize(">L")
    count = 0
    

    while True:
        # ������ ����
        while len(data) < payload_size:
            data += client_socket.recv(4096)
            
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += client_socket.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        print("@@@@@@recv success@@@@@@ {}".format(msg_size)) # ������ ũ�� ���

        # ������ȭ(de-serialization) : ����ȭ�� �����̳� ����Ʈ�� ������ ��ü�� �����ϴ� ��
        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes") # ����ȭ�Ǿ� �ִ� binary file�� ���� ��ü�� ������ȭ
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR) # ������ ���ڵ�

        # ���� ���
        cv2.imshow('img',frame)
        cv2.imwrite(str(count)+'.jpg',frame)
        count+=1
        if cv2.waitKey(1) == ord('q') : # q�� �Է��ϸ� ����
            cv2.destroyWindow('img')

        time.sleep(3)
        cv2.destroyWindow('img')
        


if __name__ == "__main__":
    
    ip = '114.70.23.41' # ip �ּ�
    port = 5050 # port ��ȣ

    # ���� ��ü�� ���� �� ����
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ip, port))
    print('���� ����')

    recv_img(client_socket)

