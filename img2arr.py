
import sys
import cv2

matimg = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
imagename = sys.argv[1][:-4]
print(imagename)
textname = imagename + ".txt"

h, w = matimg.shape

with open(textname, 'w') as file:
    print(textname)
    file.write(str(h) + ' ' + str(w) + '\n')
    for i in range(0, h):
        for j in range(0, w):
            file.write(str(float(matimg[i, j]/255.0)) + '\n')
