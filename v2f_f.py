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
import imutils
import csv
from os import walk
import pandas as pd

path =r'C:\ADITYA\Computervision\UCF50_videos\UCF50\TrampolineJumping'
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

i=0
path1 =r'C:\ADITYA\Computervision\UCF50_videos\UCF50\TrampolineJumping\\'
for dirpath,dirnames,filename in walk(path):
    # do your stuff

#for filename in glob.glob(os.path.join(path, '*.txt')):
    
    while i <=118 :#<= len(filename):
        vidcap=cv2.VideoCapture(os.path.join(path, filename[i]))
        success,image = vidcap.read()
        count=0
        while success:
            cv2.imwrite(r"C:\ADITYA\Computervision\UCF50_videos\BaseballPitch_frames\frame%d.jpg" % count, image)     # save frame as JPEG file      
            success,image = vidcap.read()
            print('Read a new frame:%d '%count, success)
            count += 1
            print(i)
            
        imagePaths = list(paths.list_images(args["images"]))
        d=0

        for imagePath in imagePaths:
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=min(400, image.shape[1]))
            orig = image.copy()
  
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
            
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show some information on the number of bounding boxes
           # filename = imagePath[imagePath.rfind("/") + 1:]
           # print("[INFO] {}: {} original boxes, {} after suppression".format(
            #        filename, len(rects), len(pick)))
            cv2.imshow("Before NMS", orig)
            cv2.imshow("After NMS", image)
            path2=r'C:\ADITYA\Computervision\UCF50_videos\BaseballPitch_roi\file_%d.jpg'%d
    #cv2.imwrite(os.path.join(path1 , 'waka.jpg'), image)
    #filename = "images/file_%d.jpg"%d
            cv2.imwrite(path2, image)
            print("frame_%d"%d)
    #a=np.array(value.flatten())
    
    #with open("Output.csv","w") as Output_csv:
     #   CSVWriter = csv.writer(Output_csv)
      #  CSVWriter.writerow(a)
            print(rects)
   # print(a)
            
            f_handle = open(r"C:\ADITYA\Computervision\UCF50_videos\UCF50_WORK\TrampolineJumpingROI\%s_roi.csv"%filename[i], 'a')
            np.savetxt(f_handle,rects,fmt='%d')
            f_handle.close()   

            if len(rects)!=0:
                
                crop_img = image[y:y+h, x:x+w]
                cv2.imshow("cropped", crop_img) 
                cv2.imwrite(path2,crop_img)
                img = cv2.imread(path2,0)     
                dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)

                mag_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
                A=cv2.cartToPolar(dft_shift[:,:,0],dft_shift[:,:,1]);
                magnitude_spectrum=20*np.log(A[0]);
                Phase_spectrum=A[1];  
                print(i)
                with open(r'C:\ADITYA\Computervision\UCF50_videos\UCF50_WORK\TrampolineJumpingfft\%s.csv'%filename[i],'a',newline='') as out:
                    csv_out=csv.writer(out)
                    csv_out.writerow(A) 
            d+=1
    
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        folder=r'C:\ADITYA\Computervision\UCF50_videos\BaseballPitch_roi'
 
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):            os.unlink(file_path)                    
            except Exception as e:
                print(e)                
        folder1=r'C:\ADITYA\Computervision\UCF50_videos\BaseballPitch_frames'
        for the_file in os.listdir(folder1):
            file_path = os.path.join(folder1, the_file)
            try:
                if os.path.isfile(file_path):            os.unlink(file_path)                    
            except Exception as e:
                print(e)                 
        i+=1