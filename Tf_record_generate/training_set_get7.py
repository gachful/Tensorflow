# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:06:35 2017

@author: yu
"""
import tensorflow as tf
import numpy as np
import json as js
from PIL import Image
from PIL import ImageDraw
import pickle
annoJson=js.load(open('E:/Yu/data_backup/coco/annotations/instances_val2014.json'))
anno2014_o=annoJson['annotations']

anno2014_index=[i for i,_ in enumerate(anno2014_o) if _["image_id"]<=400000 ]
anno2014=[anno2014_o[i] for i in anno2014_index]
#%%
def dataget():
    global imgSeg
    global target_cate_data
    while True:
        temp_id=np.random.randint(0,len(anno2014))
        temp_data=anno2014[temp_id]
        try:            
            regionSeg=temp_data['segmentation']
            regionfill=tuple(regionSeg[0])
            break
        except:
            continue
        
    
    temp_cate_id=temp_data['category_id']
    temp_pic_id=temp_data['image_id']
    temp_pic_id1=str(temp_pic_id)
    temp_pic_id2=temp_pic_id1.zfill(12)
    temp_bbox=temp_data['bbox'] 
    
    
    pic_file="E:/Yu/data_backup/coco/images/COCO_val2014_%s.jpg" % (temp_pic_id2)
    temp_pic=Image.open(pic_file)
    
    
    
    region_pic=Image.new("RGB",temp_pic.size,"black")
    draw=ImageDraw.Draw(region_pic)
   
    draw.polygon(regionfill,fill=(255,255,255))
    samecat=np.random.randint(0,5)
    if samecat<=1:
        while True:
            target_id=np.random.randint(0,len(anno2014))
            target_data=anno2014[target_id]
            try:            
                regionSeg=target_data['segmentation']
                regionfill=tuple(regionSeg[0])
                break
            except:
                continue
    else:
        cate_index=[i for i,_ in enumerate(anno2014) if _["category_id"]==temp_cate_id ]
        img_cate=[anno2014[i] for i in cate_index]
    #    
        target_id=np.random.randint(0,len(img_cate))
        target_data=img_cate[target_id]
    
    target_pic_id=target_data['image_id']
    target_pic_id1=str(target_pic_id)
    target_pic_id2=target_pic_id1.zfill(12)
    target_bbox=target_data['bbox'] 
    pic_file2="E:/Yu/data_backup/coco/images/COCO_val2014_%s.jpg" % (target_pic_id2)
    target_pic=Image.open(pic_file2)
    cate2_index=[i for i,_ in enumerate(anno2014) if _["image_id"]==target_pic_id if _["category_id"]==temp_cate_id]
    
    target_cate_data=[anno2014[i] for i in cate2_index]
    label_pic=Image.new("RGB",target_pic.size,"black")
    draw=ImageDraw.Draw(label_pic)
    for i in range(len(target_cate_data)):
        imgSeg=target_cate_data[i]['segmentation']        
        try:
            b=tuple(imgSeg[0])
            draw.polygon(b,fill=(255,255,255))
        except:
            continue
   
    return target_pic,temp_pic,region_pic,label_pic
def _bytes_feature(value):  
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  
def ima2array(imainput):
    a=np.array(imainput.size)
    b=np.array([3])
    c=np.append(a[::-1],b)   
    arroutput=np.array(imainput.getdata(),dtype='float32')    
    arroutput=np.reshape(imainput,c)/255
    return arroutput
for i in range(0,1000):
    print(i)
    writer= tf.python_io.TFRecordWriter("E:/Yu/data_backup/coco_template_tf_4/coco%s.tfrecords"%i) #要生成的文件

    for j in range(1000):
        print(j)
        try:
            target_pic,temp_pic,region_pic,label_pic=dataget()
            target_pic=target_pic.resize((128,128))
            temp_pic=temp_pic.resize((128,128))
            region_pic=region_pic.resize((128,128))
            label_pic=label_pic.resize((128,128))
            target_1=ima2array(target_pic)
            temp_1=ima2array(temp_pic)
            region_1=ima2array(region_pic)
            region_1=np.mean(region_1,2,keepdims=True)
            label_1=ima2array(label_pic)
            label_1=np.mean(label_1,2,keepdims=True)
            output1=np.concatenate((target_1,temp_1,region_1),2)
            output2=label_1
            example = tf.train.Example(features=tf.train.Features(feature={  
                'image_raw': _bytes_feature(output1.tobytes()),
                'label': _bytes_feature(output2.tobytes()) 
            })) 
        except:
            continue
        writer.write(example.SerializeToString())  #序列化为字符串
        
        
     

 


    
    
    





