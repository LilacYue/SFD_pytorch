from s3fd.face_detector import face_detector
import imageio
import cv2
import time

fd = face_detector()
reader=imageio.get_reader("./data/test01.mp4")
fps = reader.get_meta_data()['fps']

writer = imageio.get_writer('./data/test01_output.mp4', fps=fps, macro_block_size=None)

total_time = 0.0
for i, im in enumerate(reader):
    img = im[:,:,::-1]
    # import ipdb; ipdb.set_trace()
    t1 = time.clock()
    bboxlist = fd.detect_face(img)
    total_time += time.clock() - t1
    i += 1
    print("{} faces detected in frame # {}".format(len(bboxlist), i))
    for b in bboxlist:
        x1,y1,x2,y2,s = b
        cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1)
    writer.append_data(im,)
   
print("fps = ", (i+1)/total_time)
writer.close()

