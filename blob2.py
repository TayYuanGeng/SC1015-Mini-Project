import cv2
import os
import numpy as np;
import csv

class Blob:
    def __init__(self, size):
        self.size = size
        self.avg_b, self.avg_g, self.avg_r, self.avg_h, self.avg_s, self.avg_v = 0,0,0,0,0,0
        self.std_b, self.std_g, self.std_r, self.std_h, self.std_s, self.std_v = 0,0,0,0,0,0
        self.wavg_multiplier = 0.0
        
    def __enum__(self, num:int):
        return (["size", "avg_b", "avg_g", "avg_r", "avg_h", "avg_s", "avg_v", "std_b", "std_g", "std_r", "std_h", "std_s", "std_v", "wavg_multiplier"][num])
    
    def __iter__(self):
          for each in self.__dict__.values():
              yield each
        
       
# # directory only containing pictures
directory = os.path.dirname(os.path.realpath(__file__)) + "/dataset" #MacOS probs
# directory = "./dataset"

# #creates a new dir for formatted pictures
# save_dir = directory + "\\formatted"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# define range of fire color in HSV
LOWER_FIRE = np.array([0,50,50])
UPPER_FIRE = np.array([25,255,255])
SIZE_FILTER = 5 #in pixels


def unsharp_mask(img, blur_size = (5,5), imgWeight = 1.5, gaussianWeight = -0.5):
    gaussian = cv2.GaussianBlur(img, blur_size, 0)
    return cv2.addWeighted(img, imgWeight, gaussian, gaussianWeight, 0)


def smoother_edges(img, first_blur_size, second_blur_size = (5,5), imgWeight = 1.5, gaussianWeight = -0.5):
    img = cv2.GaussianBlur(img, first_blur_size, 0)
    return unsharp_mask(img, second_blur_size, imgWeight, gaussianWeight)


def close_image(img, size = (5,5)):
    kernel = np.ones(size, np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def open_image(img, size = (5,5)):
    kernel = np.ones(size, np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def shrink_rect(rect, scale = 0.8):
    center, (width, height), angle = rect
    width = width * scale
    height = height * scale
    rect = center, (width, height), angle
    return rect


def clahe(img, clip_limit = 2.0):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(5,5))
    return clahe.apply(img)


def get_sobel(img, size = -1):
    sobelx64f = cv2.Sobel(img,cv2.CV_64F,2,0,size)
    abs_sobel64f = np.absolute(sobelx64f)
    return np.uint8(abs_sobel64f)

def bgr_to_hsv(bgr_val):
    b, g, r = bgr_val.item(0)/255.0, bgr_val.item(1)/255.0, bgr_val.item(2)/255.0
    mx = max(b, g, r)
    mn = min(b, g, r)
    df = mx-mn
    hsv_val = [0,0,0]
    if mx == mn:
        hsv_val[0] = 0
    elif mx == r:
        hsv_val[0] = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        hsv_val[0] = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        hsv_val[0] = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        hsv_val[1] = 0
    else:
        hsv_val[1] = (df/mx)*100
    hsv_val[2] = mx*100
    return hsv_val
        
def get_variables (pic):
    hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_FIRE, UPPER_FIRE)
    result = smoother_edges(mask, (5,5))
    total_area = 0
    all_blobs = []

    ret, thresh = cv2.threshold(result, 127,255,cv2.THRESH_BINARY)
    contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blob_count = len(contours)
    #print("total number of blobs: " + str(blob_count))


    contour_mask = np.zeros(pic.shape[:2], np.uint8)
    # cv2.drawContours(contour_mask, [contours[205]], -1, (255,255,255), cv2.FILLED)

    for i, cnt in enumerate(contours):
        mask2 = np.zeros(pic.shape[:2], np.uint8)
        cv2.drawContours(mask2, [cnt], 0, (255,255,255), cv2.FILLED)
        #print(cv2.cvtColor(cv2.mean(pic, mask=mask2)[:3], cv2.COLOR_BGR2HSV))

        blob_output = Blob(cv2.countNonZero(mask2))
        
        
        if (blob_output.size < SIZE_FILTER):
            continue
        
        total_area += blob_output.size
        bgr_mean, bgr_stddev = cv2.meanStdDev(pic, mask=mask2)
        bgr_mean = bgr_mean[:3]
        bgr_stddev = bgr_stddev[:3]
        
        blob_output.avg_b = bgr_mean.item(0)
        blob_output.avg_g = bgr_mean.item(1)
        blob_output.avg_r = bgr_mean.item(2)
        hsv_mean = bgr_to_hsv(bgr_mean)
        blob_output.avg_h = hsv_mean[0]
        blob_output.avg_s = hsv_mean[1]
        blob_output.avg_v = hsv_mean[2]
        
        blob_output.std_b = bgr_stddev.item(0)
        blob_output.std_g = bgr_stddev.item(1)
        blob_output.std_r = bgr_stddev.item(2)
        hsv_stddev = bgr_to_hsv(bgr_stddev)
        blob_output.std_h = hsv_stddev[0]
        blob_output.std_s = hsv_stddev[1]
        blob_output.std_v = hsv_stddev[2]
        
        all_blobs.append(blob_output)
        cv2.drawContours(contour_mask, [cnt], 0, (255,255,255), cv2.FILLED)

    for j in all_blobs:
        j.wavg_multiplier = j.size / total_area
    
    return (all_blobs, blob_count)

def process_variables (input:list):
    all_blobs:list
    all_blobs, blob_count = input
    output = [0] * 14
    for x in all_blobs:
        x:Blob
        for i, y in enumerate(x):
            output[i] += y * x.wavg_multiplier
        output[13] = blob_count
    return output
        
with open(os.path.dirname(os.path.realpath(__file__)) + "/output/output_stats.csv", "w", newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(["image_name", "size", "avg_b", "avg_g", "avg_r", "avg_h", "avg_s", "avg_v", "std_b", "std_g", "std_r", "std_h", "std_s", "std_v", "blob_count", "fire"])
    length = len(os.listdir(directory))
    for i, file in enumerate(os.listdir(directory)):
        print ("File %d of %d" % (i+1, length) )
        if(file.endswith(".png") or file.endswith(".jpg")):
            try:
                pic = cv2.resize(cv2.imread(rf"{directory}/{file}"), (512, 512), interpolation = cv2.INTER_NEAREST)
                writer.writerow([file, *process_variables(get_variables(pic)), 0 if file.find("non_fire") != -1 else 1])
            except:
                print(file + " is invalid format!")
                continue