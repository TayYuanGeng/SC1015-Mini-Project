import cv2
import os
import numpy as np;

# # directory only containing pictures
# directory = r"C:\test"

# #creates a new dir for formatted pictures
# save_dir = directory + "\\formatted"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# define range of fire color in HSV
lower_fire = np.array([0,50,50])
upper_fire = np.array([25,255,255])


# params = cv2.SimpleBlobDetector_Params()
 
# # Change thresholds
# params.minThreshold = 10
# params.maxThreshold = 2000
 



# detector = cv2.SimpleBlobDetector_create(params)



def unsharp_mask(img, blur_size = (5,5), imgWeight = 1.5, gaussianWeight = -0.5):
    gaussian = cv2.GaussianBlur(img, (5,5), 0)
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


# pic = cv2.resize(cv2.imread(r"C:\test\blobs4.jpg"), (512, 512), interpolation = cv2.INTER_NEAREST)

# cv2.imshow("Keypoints", smoother_edges(pic, (9,9)))
# cv2.waitKey(0)
# cv2.imshow("Keypoints", pic)
# cv2.waitKey(0)


#file = cv2.resize(cv2.imread(r"C:\test\BlobTest.jpg"), (512, 512), interpolation = cv2.INTER_NEAREST)

#pic = cv2.resize(cv2.imread(r"C:\test\fire.jpg"), (512, 512), interpolation = cv2.INTER_NEAREST)



#TODO: get average value per blob, compute weighted average for size of blob

pic = cv2.imread(r"C:\test\article.jpg")
hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_fire, upper_fire)
result_c = cv2.bitwise_and(pic, pic, mask=mask)
result = cv2.bitwise_not(mask)

black = smoother_edges(result, (5,5))

cv2.imshow("test", black)
ret, thresh = cv2.threshold(black, 1, 1, 1)
contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
contour_mask = np.zeros(pic.shape, dtype = np.uint8)
contour_mask = cv2.bitwise_xor(pic, pic)
cv2.drawContours(contour_mask, [contours[205]], -1, (255,255,255), cv2.FILLED)

cv2.imshow("one", contour_mask)
# for cnt in contours:
    
#     cv2.drawContours(pic, [cnt], 0, (250,0,250), cv2.LINE_4)
    
    

cv2.imshow('img', pic)
cv2.waitKey(0)
cv2.destroyAllWindows()





# for file in os.listdir(directory):
#     if(file.endswith(".png") or file.endswith(".jpg")):
#         pic = cv2.resize(cv2.imread(rf"{directory}\{file}"), (512, 512), interpolation = cv2.INTER_NEAREST)
#         hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
#         # cv2.imshow("aa",hsv)
#         # cv2.waitKey(0)
#         mask = cv2.inRange(hsv, lower_fire, upper_fire)
#         result_c = cv2.bitwise_and(pic, pic, mask=mask)
#         result = cv2.bitwise_not(mask)
        
#         keypoints = detector.detect(result)
        

#         im_with_keypoints = cv2.drawKeypoints(result, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#         cv2.imshow("Keypoints", im_with_keypoints)
#         cv2.waitKey(0)

#         # filename = f'{file.replace(".png","")}_blob.png'
#         # os.chdir(save_dir)
#         # cv2.imwrite(filename,result)



