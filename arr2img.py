
import sys
import cv2
import numpy

txtname = sys.argv[1][:-4]
show = False
if len(sys.argv) == 3 and sys.argv[2] == "--show":
    show = True
imagename = txtname + ".png"
print(imagename)


with open(sys.argv[1], 'r') as file:
    print(txtname)
    s = file.readline().split(" ")
    h, w = int(s[0]), int(s[1])
    print(h, w)
    img = numpy.zeros([h,w])
    for i in range(0, h):
        for j in range(0, w):
            img[i, j] = float(file.readline())


    img *= 255
    cv2.imwrite(imagename, img)
    if show:
        wroteimg = cv2.imread(imagename)
        cv2.imshow("wrote img", wroteimg)
        cv2.waitKey()
