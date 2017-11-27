from s3fd.face_detector import face_detector
import cv2

fd = face_detector()
cap=cv2.VideoCapture(0)

ret, frame = cap.read()

while True:
    if ret:
        ret, img = cap.read()
        imgshow = np.copy(img)
        bboxlist = fd.detect_face(img)
        for b in bboxlist:
            x1,y1,x2,y2,s = b
            cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1)
        cv2.imshow('test',imgshow)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    else:
        print("something wrong with the webcam")
        break
 
