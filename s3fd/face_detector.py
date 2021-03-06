from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime,time,math
import argparse
import numpy as np

from .net_s3fd import s3fd as Net
from .bbox import *
pwd = os.path.dirname(os.path.abspath(__file__))

class face_detector(object):
    def __init__(self):
        self.net = Net()
        self.net.load_state_dict(torch.load(pwd + '/s3fd_convert.pth'))
        use_cuda = torch.cuda.is_available()
        self.net.cuda()
        self.net.eval()
    def detect_face(self,img):
        # print("on detect_face")
        img = img - np.array([104,117,123])
        img = img.transpose(2, 0, 1)
        img = img.reshape((1,)+img.shape)

        img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
        BB,CC,HH,WW = img.size()
        olist = self.net(img)

        bboxlist = []
        for i in range(int(len(olist)/2)): olist[i*2] = F.softmax(olist[i*2])
        for i in range(int(len(olist)/2)):
            ocls,oreg = olist[i*2].data.cpu(),olist[i*2+1].data.cpu()
            FB,FC,FH,FW = ocls.size() # feature map size
            stride = 2**(i+2)    # 4,8,16,32,64,128
            anchor = stride*4
            for Findex in range(FH*FW):
                windex,hindex = Findex%FW,Findex//FW
                axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
                score = ocls[0,1,hindex,windex]
                loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
                if score<0.05: continue
                priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]])
                variances = [0.1,0.2]
                box = decode(loc,priors,variances)
                x1,y1,x2,y2 = box[0]*1.0
                # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
                bboxlist.append([x1,y1,x2,y2,score])
        bboxlist = np.array(bboxlist)
        if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
        keep = nms(bboxlist,0.3)
        bboxlist = bboxlist[keep,:]
        bboxlist_final = []
        for b in bboxlist:
            x1,y1,x2,y2,s = b
            if s<0.5: continue
            bboxlist_final.append(b)
        return np.asarray(bboxlist_final)

    def read_img(self,path_to_img):
        return cv2.imread(path_to_img)
     

