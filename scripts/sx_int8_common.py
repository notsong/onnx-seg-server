"""Shared SX test preprocessing and annotation metrics; matches the C++ service."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.stderr.reconfigure(encoding='utf-8',errors='replace')
sys.path[:0]=[str(ROOT/'.int8-test-deps'),str(ROOT/'.test-deps'),str(ROOT/'.openvino-test-deps')]
import cv2
import numpy as np
from PIL import Image,ImageDraw

def preprocess(path):
    image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('Cannot decode '+str(path))
    h,w=image.shape[:2]
    scale=1024/max(h,w)
    nh,nw=max(1,int(h*scale)),max(1,int(w*scale))
    prepared=np.full((1024,1024,3),128,dtype=np.uint8)
    top,left=(1024-nh)//2,(1024-nw)//2
    prepared[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_LINEAR)
    blob=cv2.dnn.blobFromImage(prepared,1.0/255.0,(1024,1024),swapRB=True,crop=False)
    for c,(mean,std) in enumerate(zip([0.485,0.456,0.406],[0.229,0.224,0.225])):
        blob[0,c]-=np.float32(mean)
        blob[0,c]/=np.float32(std)
    return blob,(w,h),(left,top,nw,nh)

def truth(path):
    d=json.loads(path.with_suffix('.json').read_text(encoding='utf-8-sig'))
    mask=Image.new('L',(d['imageWidth'],d['imageHeight']),0)
    draw=ImageDraw.Draw(mask)
    for shape in d['shapes']:
        if shape['label']!='sx' or shape['shape_type']!='polygon':
            raise ValueError('Unsupported annotation')
        draw.polygon([tuple(p) for p in shape['points']],fill=1)
    return np.asarray(mask,dtype=bool)

def metrics(pred,gt):
    tp=int(np.count_nonzero(pred & gt));fp=int(np.count_nonzero(pred & ~gt))
    fn=int(np.count_nonzero(~pred & gt));tn=int(np.count_nonzero(~pred & ~gt))
    return {'iou':tp/(tp+fp+fn),'dice':2*tp/(2*tp+fp+fn),
            'pixel_accuracy':(tp+tn)/gt.size,'tp':tp,'fp':fp,'fn':fn,'tn':tn}
