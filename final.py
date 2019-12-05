# USAGE
# python detect.py --images images
# import the necessary packages
from __future__ import print_function
from __future__ import with_statement
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from imutils.object_detection import non_max_suppression
from imutils import paths
import os, shutil
import argparse
#import os
import imutils
import csv
import pandas as pd

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#video capture
vidcap = cv2.VideoCapture(r'C:\ADITYA\Computervision\UCF50_videos\UCF50\BaseballPitch\v_BaseballPitch_g01_c01.avi')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(r"C:\ADITYA\Computervision\UCF50_videos\BaseballPitch_frames\frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

# loop over the image paths
imagePaths = list(paths.list_images(args["images"]))
d=0

for imagePath in imagePaths:
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
    ###############
   # value=np.array(image)
   # im=Image.open(imagePath)
   # pix=im.load()
   # width, height = im.size
    
    #######################
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
     padding=(8, 8), scale=1.05)
    
	# draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    
	# draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show some information on the number of bounding boxes
    filename = imagePath[imagePath.rfind("/") + 1:]
    print("[INFO] {}: {} original boxes, {} after suppression".format(
            filename, len(rects), len(pick)))

	# show the output images
    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    path1=r'C:\ADITYA\Computervision\UCF50_videos\BaseballPitch_roi\file_%d.jpg'%d
    #cv2.imwrite(os.path.join(path1 , 'waka.jpg'), image)
    #filename = "images/file_%d.jpg"%d
    cv2.imwrite(path1, image)
    print("frame_%d"%d)
    #a=np.array(value.flatten())
    
    #with open("Output.csv","w") as Output_csv:
     #   CSVWriter = csv.writer(Output_csv)
      #  CSVWriter.writerow(a)
    print(rects)
   # print(a)
    #with open(r"E:\Computervision\UCF50_videos\UCF50\vid1.csv"+'.x.betas','a') as f_handle:
    #   np.savetxt(f_handle,rects)
    crop_img = image[y:y+h, x:x+w]
    cv2.imshow("cropped", crop_img) 
    cv2.imwrite(path1,crop_img)
    
    img = cv2.imread(path1,0)

    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mag_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    A=cv2.cartToPolar(dft_shift[:,:,0],dft_shift[:,:,1]);
    magnitude_spectrum=20*np.log(A[0]);
    Phase_spectrum=A[1];
    
    with open('vid_fft1.csv','a',newline='') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(A)
   # f_handle1 = open('vid1_fft1.csv','a')
   # np.savetxt(f_handle1,A[0],fmt='%d')
   # f_handle1.close()


    f_handle = open(r"C:\ADITYA\Computervision\UCF50_videos\UCF50\vid1.csv", 'a')
    #np.savetxt(f_handle,"frame%d",d)
    np.savetxt(f_handle,rects,fmt='%d')
    f_handle.close()   
    d+=1
    #with open('Fail.csv', 'wb') as csvfile:
    #writer = csv.writer(csvfile, delimiter=',')
    #writer.writerow(('some_info','other_info',image.flatten()))
    
    
#    with open('output_file.csv', 'w+') as f:
#        f.write('R,G,B\n')
 
  #read the details of each pixel and write them to the file
#        for x in range(width):
#            for y in range(height):
#                r = pix[x,y][0]
#                g = pix[x,x][1]
#                b = pix[x,x][2]
#                f.write('{0},{1},{2}\n'.format(r,g,b))
    cv2.waitKey(1)
    cv2.destroyAllWindows()

#Clearing the directory
folder=r'C:\ADITYA\Computervision\UCF50_videos\BaseballPitch_roi'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
       if os.path.isfile(file_path):            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)
        