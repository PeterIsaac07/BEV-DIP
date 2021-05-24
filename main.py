import os
import sys
import socket
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

calibration_path = 'calibration_params'
if not os.path.isdir(calibration_path):
    os.mkdir(calibration_path)

frames_dir = 'frames'
if not os.path.isdir(frames_dir):
    os.mkdir(frames_dir)
right_dir = 'frames/right'
if not os.path.isdir(right_dir):
    os.mkdir(right_dir)
left_dir = 'frames/left'
if not os.path.isdir(left_dir):
    os.mkdir(left_dir)
front_dir = 'frames/front'
if not os.path.isdir(front_dir):
    os.mkdir(front_dir)
back_dir = 'frames/back'
if not os.path.isdir(back_dir):
    os.mkdir(back_dir)
    


calibration_frames_dir = 'calibration_frames'
if not os.path.isdir(calibration_frames_dir):
    os.mkdir(calibration_frames_dir)
right_dir_calib = 'calibration_frames/right'
if not os.path.isdir(right_dir_calib):
    os.mkdir(right_dir_calib)
left_dir_calib = 'calibration_frames/left'
if not os.path.isdir(left_dir_calib):
    os.mkdir(left_dir_calib)
front_dir_calib = 'calibration_frames/front'
if not os.path.isdir(front_dir_calib):
    os.mkdir(front_dir_calib)
back_dir_calib = 'calibration_frames/back'
if not os.path.isdir(back_dir_calib):
    os.mkdir(back_dir_calib)


rightpath = os.path.join(frames_dir,'right')
leftpath = os.path.join(frames_dir,'left')
frontpath = os.path.join(frames_dir,'front')
backpath = os.path.join(frames_dir,'back')

rightpath_calib = os.path.join(calibration_frames_dir,'right')
leftpath_calib = os.path.join(calibration_frames_dir,'left')
frontpath_calib = os.path.join(calibration_frames_dir,'front')
backpath_calib = os.path.join(calibration_frames_dir,'back')



calibration_file_path_list = []
calibration_params_file = 'calibration_params_'+'left'+'.pickle'
calibration_file_path_left = os.path.join(calibration_path,calibration_params_file)
calibration_file_path_list.append(calibration_file_path_left)
calibration_params_file = 'calibration_params_'+'right'+'.pickle'
calibration_file_path_right = os.path.join(calibration_path,calibration_params_file)
calibration_file_path_list.append(calibration_file_path_right)
calibration_params_file = 'calibration_params_'+'front'+'.pickle'
calibration_file_path_front = os.path.join(calibration_path,calibration_params_file)
calibration_file_path_list.append(calibration_file_path_front)
calibration_params_file = 'calibration_params_'+'back'+'.pickle'
calibration_file_path_back = os.path.join(calibration_path,calibration_params_file)
calibration_file_path_list.append(calibration_file_path_back)




HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            conn.sendall(data)

def readnbyte(sock, n):
    buff = bytearray(n)
    pos = 0
    while pos < n:
        cr = sock.recv_into(memoryview(buff)[pos:])
        pos += cr
    return buff

#Reconstructing n bytes received to images
def read_TCP_image(data):
    imagenumpy = np.array(data,dtype = np.uint8)
    R1 = imagenumpy[0:Res].reshape((H,W))
    G1 = imagenumpy[Res:Res*2].reshape((H,W))
    B1 = imagenumpy[Res*2:Res*3].reshape((H,W))
    imgL = np.dstack((R1,G1,B1))
    imgL = np.rot90(imgL, 3)
    return imgL

#Reads images and saves them or shows them
def get_frames_four_sides(path,sizeimg,num_frames = -1,port = 1117,host = "127.0.0.1",save_img = True , show_img = False):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        conn, addr = s.accept()
        iterator2 = 0
        with conn:
            print('Connected by', addr)
            while True:
                data = readnbyte(conn,sizeimg*4)
                if (not data):
                    cv2.destroyAllWindows()
                    break
                imgL = read_TCP_image(data[0:sizeimg])
                imgR = read_TCP_image(data[sizeimg:sizeimg*2])
                imgT = read_TCP_image(data[sizeimg*2:sizeimg*3])
                imgB = read_TCP_image(data[sizeimg*3:sizeimg*4])
                if(save_img):
                    cv2.imwrite(os.path.join(path,'left/frame'+str(iterator2)+'.png'),imgL)
                    cv2.imwrite(os.path.join(path,'right/frame'+str(iterator2)+'.png'),imgR)
                    cv2.imwrite(os.path.join(path,'front/frame'+str(iterator2)+'.png'),imgT)
                    cv2.imwrite(os.path.join(path,'back/frame'+str(iterator2)+'.png'),imgB)
                if(show_img): 
                    cv2.imshow('frame',imgL)
                    cv2.waitKey(1)
                iterator2+= 1
                if (iterator2 == num_frames):
                    break



def calibrate_fish(chess_board_dims,calib_dir):
    CHECKERBOARD = chess_board_dims
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    calibrationdir = calib_dir
    for fname in os.listdir(calibrationdir):
        img = cv2.imread(os.path.join(calibrationdir,fname))
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        #print (ret,corners)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print(fname)
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    DIM = _img_shape[::-1]
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    return K,D,DIM



def calibrate_and_save(frames_path,calibration_file_path,cardboard_dims=7):
    chess_board_dims = (cardboard_dims,cardboard_dims)
    calibration_params = calibrate_fish(chess_board_dims,frames_path)
    calibration_file = open(calibration_file_path, "wb")
    pickle.dump(calibration_params,calibration_file)
    calibration_file.close()

def load_calibration_params(calibration_file_path):
    calibration_file = open(calibration_file_path, "rb")
    calibration_params = pickle.load(calibration_file)
    calibration_file.close()
    return calibration_params