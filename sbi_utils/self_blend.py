from glob import glob
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import shutil
import json
import sys
import argparse
import dlib
from imutils import face_utils
import albumentations as alb
import blend as B
from initialize import *
from funcs import IoUfrom2bboxes,crop_face,RandomDownScale
import logging
import warnings
warnings.filterwarnings('ignore')

if os.path.isfile('./src/utils/library/bi_online_generation.py'):
	sys.path.append('./src/utils/library/')
	print('exist library')
	exist_bi=True
else:
	exist_bi=False

if exist_bi:
    from library.bi_online_generation import random_get_hull

def get_transforms():
    return alb.Compose([
        alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
        alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
        alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
        alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
    ], 
    additional_targets={f'image1': 'image'},
    p=1.)


def get_source_transforms():
    return alb.Compose([
            alb.Compose([
                    alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
                    alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=1),
                    alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=1),
                ],p=1),

            alb.OneOf([
                RandomDownScale(p=1),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ],p=1),
            
        ], p=1.)

def randaffine(img,mask):
    f=alb.Affine(
            translate_percent={'x':(-0.03,0.03),'y':(-0.015,0.015)},
            scale=[0.95,1/0.95],
            fit_output=False,
            p=1)
        
    g=alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

    transformed=f(image=img,mask=mask)
    img=transformed['image']
    
    mask=transformed['mask']
    transformed=g(image=img,mask=mask)
    mask=transformed['mask']
    return img,mask


transforms=get_transforms()
source_transforms = get_source_transforms()
def self_blending(img,landmark):
    H,W=len(img),len(img[0])
    if np.random.rand()<0.25:
        landmark=landmark[:68]
    if exist_bi:
        logging.disable(logging.FATAL)
        mask=random_get_hull(landmark,img)[:,:,0]
        logging.disable(logging.NOTSET)
    else:
        mask=np.zeros_like(img[:,:,0])
        cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)


    source = img.copy()
    if np.random.rand()<0.5:
        source = source_transforms(image=source.astype(np.uint8))['image']
    else:
        img = source_transforms(image=img.astype(np.uint8))['image']

    source, mask = randaffine(source,mask)

    img_blended,mask=B.dynamic_blend(source,img,mask)
    img_blended = img_blended.astype(np.uint8)
    img = img.astype(np.uint8)

    return img,img_blended,mask


def reorder_landmark(landmark):
    landmark_add=np.zeros((13,2))
    for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
        landmark_add[idx]=landmark[idx_l]
    landmark[68:]=landmark_add
    return landmark

def hflip(img,mask=None,landmark=None,bbox=None):
    H,W=img.shape[:2]
    landmark=landmark.copy()
    bbox=bbox.copy()

    if landmark is not None:
        landmark_new=np.zeros_like(landmark)

        
        landmark_new[:17]=landmark[:17][::-1]
        landmark_new[17:27]=landmark[17:27][::-1]

        landmark_new[27:31]=landmark[27:31]
        landmark_new[31:36]=landmark[31:36][::-1]

        landmark_new[36:40]=landmark[42:46][::-1]
        landmark_new[40:42]=landmark[46:48][::-1]

        landmark_new[42:46]=landmark[36:40][::-1]
        landmark_new[46:48]=landmark[40:42][::-1]

        landmark_new[48:55]=landmark[48:55][::-1]
        landmark_new[55:60]=landmark[55:60][::-1]

        landmark_new[60:65]=landmark[60:65][::-1]
        landmark_new[65:68]=landmark[65:68][::-1]
        if len(landmark)==68:
            pass
        elif len(landmark)==81:
            landmark_new[68:81]=landmark[68:81][::-1]
        else:
            raise NotImplementedError
        landmark_new[:,0]=W-landmark_new[:,0]
        
    else:
        landmark_new=None

    if bbox is not None:
        bbox_new=np.zeros_like(bbox)
        bbox_new[0,0]=bbox[1,0]
        bbox_new[1,0]=bbox[0,0]
        bbox_new[:,0]=W-bbox_new[:,0]
        bbox_new[:,1]=bbox[:,1].copy()
        if len(bbox)>2:
            bbox_new[2,0]=W-bbox[3,0]
            bbox_new[2,1]=bbox[3,1]
            bbox_new[3,0]=W-bbox[2,0]
            bbox_new[3,1]=bbox[2,1]
            bbox_new[4,0]=W-bbox[4,0]
            bbox_new[4,1]=bbox[4,1]
            bbox_new[5,0]=W-bbox[6,0]
            bbox_new[5,1]=bbox[6,1]
            bbox_new[6,0]=W-bbox[5,0]
            bbox_new[6,1]=bbox[5,1]
    else:
        bbox_new=None

    if mask is not None:
        mask=mask[:,::-1]
    else:
        mask=None
    img=img[:,::-1].copy()
    return img,mask,landmark_new,bbox_new
	

def facecrop(org_path,save_path,face_detector,face_predictor,period=1,num_frames=10):
    save_path_=save_path+'frames/'+os.path.basename(org_path).replace('.mp4','/')
    for img_path in os.listdir(save_path_):
        image_path = save_path_ + img_path
        bbox_path = image_path.replace('/frames','/retina').replace('png', 'npy')
        land_path = image_path.replace('/frames','/landmarks').replace('png', 'npy')
        img = cv2.imread(image_path)
        try:
            landmark = np.load(land_path)[0]
        except:
            print(land_path)
            continue
            # import pdb; pdb.set_trace()
        bbox_lm=np.array([landmark[:,0].min(),landmark[:,1].min(),landmark[:,0].max(),landmark[:,1].max()])
        try:
            bboxes = np.load(bbox_path)[:2]
        except:
            print(bbox_path)
            continue
        iou_max = -1
        for i in range(len(bboxes)):
            iou = IoUfrom2bboxes(bbox_lm,bboxes[i].flatten())
            if iou_max < iou:
                bbox = bboxes[i]
                iou_max = iou

        try:
            landmark = reorder_landmark(landmark)
        except:
            print(land_path)
            print(landmark.shape)
            continue
            # import pdb; pdb.set_trace()
        if np.random.rand()<0.5:
            img,_,landmark,bbox = hflip(img,None,landmark,bbox)

        try:   
            img,landmark,bbox,__ = crop_face(img,landmark,bbox,margin=True,crop_by_bbox=False)
        except:
            print("crop face error")
            continue

        img_r, img_f, mask_f = self_blending(img,landmark)

        transformed = transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))
        img_f = transformed['image']
        img_r = transformed['image1']
            
        img_f, _, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(img_f, landmark, bbox, margin=False, crop_by_bbox=True, abs_coord=True, phase='train')
        
        img_r = img_r[y0_new:y1_new,x0_new:x1_new]
        
        image_size = (256,256)
        try:
            img_f = cv2.resize(img_f, image_size, interpolation=cv2.INTER_LINEAR).astype('float32')/255
            img_r = cv2.resize(img_r, image_size, interpolation=cv2.INTER_LINEAR).astype('float32')/255
        except:
            print(land_path)
            continue
            # import pdb; pdb.set_trace()

        img_f = img_f.transpose((2,0,1))
        img_r = img_r.transpose((2,0,1))
        flag=False

        # import pdb; pdb.set_trace()
        land_path_aug = land_path.replace('landmarks', 'landmarks_f')
        image_path_r = land_path.replace('landmarks', 'frames_r')
        image_path_f = land_path.replace('landmarks', 'frames_f')

        os.makedirs(os.path.dirname(land_path_aug),exist_ok=True)
        os.makedirs(os.path.dirname(image_path_r),exist_ok=True)
        os.makedirs(os.path.dirname(image_path_f),exist_ok=True)

        np.save(land_path_aug, landmark)
        np.save(image_path_r, img_r)
        np.save(image_path_f, img_f)

    return


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-d',dest='dataset',choices=['Original_taylor', 'DeepFakeDetection_original', 'DeepFakeDetection_original','DeepFakeDetection','FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures','Original','Celeb-real','Celeb-synthesis','YouTube-real','DFDC','DFDCP'])
    parser.add_argument('-c',dest='comp',choices=['raw','c23','c40'],default='raw')
    parser.add_argument('-n',dest='num_frames',type=int,default=32)
    args=parser.parse_args()
    if args.dataset=='Original':
        dataset_path='data/FaceForensics++/original_sequences/youtube/{}/'.format(args.comp)
        # dataset_path='data/original_sequences/youtube/{}/'.format(args.comp)
    elif args.dataset=='Original_taylor':
        dataset_path='data/FaceForensics-taylor/original_sequences/youtube/{}/'.format(args.comp)
    elif args.dataset=='DeepFakeDetection_original':
        dataset_path='data/FaceForensics++/original_sequences/d/{}/'.format(args.comp)
    elif args.dataset in ['DeepFakeDetection_taylor','FaceShifter_taylor','Face2Face_taylor','Deepfakes_taylor','FaceSwap_taylor','NeuralTextures_taylor']:
        dataset_path='data/FaceForensics-taylor/manipulated_sequences/{}/{}/'.format(args.dataset,args.comp)
    elif args.dataset in ['DeepFakeDetection','FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures']:
        dataset_path='data/FaceForensics++/manipulated_sequences/{}/{}/'.format(args.dataset,args.comp)
    elif args.dataset in ['Celeb-real','Celeb-synthesis','YouTube-real']:
        dataset_path='data/Celeb-DF-v2/{}/'.format(args.dataset)
    elif args.dataset in ['DFDC']:
        dataset_path='data/{}/'.format(args.dataset)
    else:
        raise NotImplementedError

    face_detector = dlib.get_frontal_face_detector()
    predictor_path = 'src/preprocess/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    
    movies_path=dataset_path+'videos/'

    movies_path_list=sorted(glob(movies_path+'*.mp4'))
    print("{} : videos are exist in {}".format(len(movies_path_list),args.dataset))


    n_sample=len(movies_path_list)

    # video_list_eval = []
    # video_list_txt='data/Celeb-DF-v2/List_of_testing_videos.txt'
    # with open(video_list_txt) as f:
    #     for data in f:
    #         tmp_path = data.split(' ')[1].split('.')[0]
    #         video_list_eval.append('data/Celeb-DF-v2/' + tmp_path.split('/')[0] + '/' + 'videos' + '/' + tmp_path.split('/')[1] + '.mp4')
    
    for i in tqdm(range(605, n_sample, 1)):
        # if movies_path_list[i] not in video_list_eval:
        #     print("not necessary")
        #     continue
        
        cap_org = cv2.VideoCapture(movies_path_list[i])
        frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
        folder_path=movies_path_list[i].replace('videos/','frames/').replace('.mp4','/')
        if len(glob(folder_path.replace('/frames/','/frames_f/')+'*'))<frame_count_org-10:
            print("processing", folder_path)
            # if os.path.exists(folder_path):
            #     print("pass ", folder_path)
            #     continue
            # else:
            #     print("processing", folder_path)
            
            # import pdb; pdb.set_trace()
            facecrop(movies_path_list[i],save_path=dataset_path,num_frames=args.num_frames,face_predictor=face_predictor,face_detector=face_detector)
    
        else:
            pass
    
