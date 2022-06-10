import numpy as np
import cv2 as cv
import os 


def makedir(): # 각 디렉토리(Case1,Case2...,Case75)에 "segmentation" 폴더를 만드는 function
    root = "open/train"
    for path in os.listdir(root):
        os.mkdir(root+'/'+path+'/'+'segmentation')

def segmentation(): # plan segmentation function
    root = "open/train"
    for path in os.listdir(root):
        sub_path = root + "/" + path
        img_path = sub_path + "/" + "image"
        for filename in os.listdir(img_path): # image 폴더에 있는 image파일들에 접근
            img = cv.imread(img_path+"/"+filename)
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) 

            # 초록색 rgb를 range로 설정하고, 다른 rgb값들을 masking
            mask = cv.inRange(hsv, (36, 0, 0), (100, 200,200))
            imask = mask>0
            green = np.zeros_like(img, np.uint8)
            green[imask] = img[imask]

            # 이미지를 흑백 컬러로 전환
            gray = cv.cvtColor(green, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

            # segmentation폴더에 저장
            cv.imwrite(sub_path + "/segmentation/" +filename,thresh)


def count_pixel():
    root = "open/train"
    ratio_list = []
    file_name = []
    cnt = 0
    for path in os.listdir(root):
        sub_path = root + "/" + path
        seg_path = sub_path + "/" + "segmentation"
        for filename in os.listdir(seg_path):
            img = cv.imread(seg_path+"/"+filename)

            sought = [255,255,255]
            white  = np.count_nonzero(np.all(img==sought,axis=2))
            sought = [0,0,0]
            black  = np.count_nonzero(np.all(img==sought,axis=2))
            total = white + black
            ratio = black / total

            dst = filename.split('.')[0]

            # print(f"filename: {dst}")
            # print(f"white: {white}")
            # print(f"black: {black}")
            # print(f"ratio of plant area: {ratio}")
            
            file_name.append(dst)
            ratio_list.append(ratio)

            cnt += 1
            if cnt % 100 == 0:
                print(cnt)
            
            
        
    with open('open/area_ratio.txt', 'w') as f:
        for i in range(len(file_name)):
            f.write("{0} ".format(file_name[i]))
            f.write("{0:0.3f} \n".format(ratio_list[i]))

# makedir()
# segmentation()
# count_pixel()



