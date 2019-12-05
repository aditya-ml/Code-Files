import numpy as np
import cv2
from matplotlib import pyplot as plt
import csv

img = cv2.imread(r'C:\ADITYA\Computervision\UCF50_videos\BaseballPitch_roi\file_79.jpg',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

mag_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
A=cv2.cartToPolar(dft_shift[:,:,0],dft_shift[:,:,1]);
magnitude_spectrum=20*np.log(A[0]);
Phase_spectrum=A[1];
print(magnitude_spectrum);
print('phase spectrum');
print(Phase_spectrum);
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.subplot(122),plt.imshow(Phase_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

with open('vid_fft1.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(A)
    #for row in A:
     #   csv_out.writerow(row)