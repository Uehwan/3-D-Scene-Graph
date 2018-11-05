import os.path as osp
import cv2
import os
from cv2 import VideoWriter, VideoWriter_fourcc
fps=30
format = "MJPG"
fourcc = VideoWriter_fourcc(*format)
RESULT_PWD = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/object_detection'
if not os.path.exists(RESULT_PWD):
    os.mkdir(RESULT_PWD)
vid = VideoWriter(osp.join(RESULT_PWD,'object_detection_result4.avi'), fourcc, float(fps), (1296,968), True)

imageFileList = sorted(os.listdir(RESULT_PWD))
for idx in range(len(imageFileList)-1):
    print(idx)
    image = cv2.imread(osp.join(RESULT_PWD, str(idx) + '.jpg'))
    #vid.write(image.astype('uint8'))
    #vid.write(image)

    cv2.imshow('frame',image)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()