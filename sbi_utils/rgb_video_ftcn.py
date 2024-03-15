# Created by: Kaede Shiohara
# Yamasaki Lab at The University of Tokyo
# shiohara@cvm.t.u-tokyo.ac.jp
# Copyright (c) 2021
# 3rd party softwares' licenses are noticed at https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE

import torch
from torchvision import datasets,transforms,utils
from torch.utils.data import Dataset,IterableDataset
from glob import glob
import os
import numpy as np
from PIL import Image
import random
import cv2
from torch import nn
import sys
import albumentations as alb
import h5py

import random
import warnings
warnings.filterwarnings('ignore')
import logging


from utils.plugin_loader import PluginLoader
import torch
import os
import numpy as np
from test_tools.common import detect_all, grab_all_frames
from test_tools.utils import get_crop_box
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter
import argparse
from tqdm import tqdm
from dataset import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import random
import torchvision

mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255,]).cuda().view(1, 3, 1, 1, 1)
std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255,]).cuda().view(1, 3, 1, 1, 1)

if os.path.isfile('/app/src/utils/library/bi_online_generation.py'):
	sys.path.append('/app/src/utils/library/')
	print('exist library')
	exist_bi=True
else:
	exist_bi=False

crop_align_func = FasterCropAlignXRay(224)


def ftcn_extract(input_file):
	basename = os.path.splitext(os.path.basename(input_file))[0] + ".avi"
	max_frame = 768
	cache_file = f"{input_file}_{str(max_frame)}.pth"

	if os.path.exists(cache_file):
		detect_res, all_lm68 = torch.load(cache_file)
		frames = grab_all_frames(input_file, max_size=max_frame, cvt=True)
		# print("detection result loaded from cache")
	else:
		# print("detecting")
		detect_res, all_lm68, frames = detect_all(
			input_file, return_frames=True, max_size=max_frame
		)
		torch.save((detect_res, all_lm68), cache_file)
		print("detect finished")

	# print("number of frames",len(frames))
	shape = frames[0].shape[:2]

	all_detect_res = []

	assert len(all_lm68) == len(detect_res)

	for faces, faces_lm68 in zip(detect_res, all_lm68):
		new_faces = []
		for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
			new_face = (box, lm5, face_lm68, score)
			new_faces.append(new_face)
		all_detect_res.append(new_faces)

	detect_res = all_detect_res

	# print("split into super clips")

	tracks = multiple_tracking(detect_res)
	tuples = [(0, len(detect_res))] * len(tracks)

	# print("full_tracks", len(tracks))

	if len(tracks) == 0:
		tuples, tracks = find_longest(detect_res)

	data_storage = {}
	frame_boxes = {}
	super_clips = []

	for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
		# print(start, end)
		assert len(detect_res[start:end]) == len(track)

		super_clips.append(len(track))

		for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
			box,lm5,lm68 = face[:3]
			big_box = get_crop_box(shape, box, scale=0.5)

			top_left = big_box[:2][None, :]

			new_lm5 = lm5 - top_left
			new_lm68 = lm68 - top_left

			new_box = (box.reshape(2, 2) - top_left).reshape(-1)

			info = (new_box, new_lm5, new_lm68, big_box)


			x1, y1, x2, y2 = big_box
			cropped = frames[frame_idx][y1:y2, x1:x2]

			base_key = f"{track_i}_{j}_"
			data_storage[base_key + "img"] = cropped
			data_storage[base_key + "ldm"] = info
			data_storage[base_key + "idx"] = frame_idx

			frame_boxes[frame_idx] = np.rint(box).astype(np.int)

	# print("sampling clips from super clips", super_clips)

	clips_for_video = []
	clip_size = 32
	pad_length = clip_size - 1

	for super_clip_idx, super_clip_size in enumerate(super_clips):
		inner_index = list(range(super_clip_size))
		if super_clip_size < clip_size: # padding
			post_module = inner_index[1:-1][::-1] + inner_index

			l_post = len(post_module)
			post_module = post_module * (pad_length // l_post + 1)
			post_module = post_module[:pad_length]
			assert len(post_module) == pad_length

			pre_module = inner_index + inner_index[1:-1][::-1]
			l_pre = len(pre_module)
			pre_module = pre_module * (pad_length // l_pre + 1)
			pre_module = pre_module[-pad_length:]
			assert len(pre_module) == pad_length
			inner_index = pre_module + inner_index + post_module

		super_clip_size = len(inner_index)

		frame_range = [
			inner_index[i : i + clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size
		]

		for indices in frame_range:
			clip = [(super_clip_idx, t) for t in indices]
			clips_for_video.append(clip)
	
	preds = []
	frame_res = {}

	for clip_idx, clip in enumerate(random.sample(clips_for_video, 10)):
		images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
		landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
		frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]
		landmarks, images = crop_align_func(landmarks, images)

		# save_path = clip[0]
		out_file = input_file.replace('++', '-clip')
		if not os.path.exist(out_file.rsplit('/', 2)[0]):
			os.makedirs(out_file.rsplit('/', 2)[0])
		
		torchvision.io.write_video(filename=out_file.split('.mp4')[0] + '_' + str(clip_idx) + '.mp4', video_array=images, fps=3, video_codec='h264')
		images = torch.as_tensor(images, dtype=torch.float32).cuda().permute(3, 0, 1, 2)
		images = images.unsqueeze(0).sub(mean).div(std)

	return images


class Video_FTCN_Dataset(Dataset):
	def __init__(self,phase='train',image_size=224,n_frames=8):
		
		assert phase in ['train','val','test']
		
		# self.data = h5py.File('data//FaceForensics++_hdf5/data_new.hdf5','r')
		image_list = np.load('data/FaceForensics++_hdf5/init_ff_video_image_list_' + phase + '.npy')
		# label_list = np.load('data/FaceForensics++_hdf5/init_ff_video_image_list_' + phase + '.npy')

		# image_list,label_list=init_ff_video_taylor(dataset='all',phase=phase)
		# image_list,label_list=init_ff(phase,'frame',n_frames=n_frames)
		
		path_lm='/landmarks/'
		# image_list=[image_list[i].replace('videos', 'frames').replace('.mp4', '') for i in range(len(image_list))]

		# label_list=[label_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/vid/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		# image_list=[image_list[i] for i in range(len(image_list)) if os.path.isfile(image_list[i].replace('/frames/',path_lm).replace('.png','.npy')) and os.path.isfile(image_list[i].replace('/frames/','/retina/').replace('.png','.npy'))]
		self.path_lm=path_lm
		print(f'Taylor({phase}): {len(image_list)}')
	
		self.image_list=image_list
		self.image_size=(image_size,image_size)
		self.phase=phase
		self.n_frames=n_frames

		# self.transforms=self.get_transforms()
		# self.source_transforms = self.get_source_transforms()

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self,idx):
		flag=True
		while flag:
			try:
				videoname=self.image_list[idx]
				imgs = ftcn_extract(videoname)[0]
				labels = []
				if 'manipulated' in videoname:
					labels.append(1)
				else:
					labels.append(0)
				imgs=np.array(imgs.cpu())
				labels=np.array(labels)
			except Exception as e:
				print(e)
				idx=torch.randint(low=0,high=len(self),size=(1,)).item()

		return imgs,labels


	def get_source_transforms(self):
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

		
	def get_transforms(self):
		return alb.Compose([
			
			alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
			alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
			alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
			alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
			
		], 
		additional_targets={f'image1': 'image'},
		p=1.)


	def randaffine(self,img,mask):
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

		
	def self_blending(self,img,landmark):
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
			source = self.source_transforms(image=source.astype(np.uint8))['image']
		else:
			img = self.source_transforms(image=img.astype(np.uint8))['image']

		source, mask = self.randaffine(source,mask)

		img_blended,mask=B.dynamic_blend(source,img,mask)
		img_blended = img_blended.astype(np.uint8)
		img = img.astype(np.uint8)

		return img,img_blended,mask
	
	def reorder_landmark(self,landmark):
		landmark_add=np.zeros((13,2))
		for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
			landmark_add[idx]=landmark[idx_l]
		landmark[68:]=landmark_add
		return landmark

	def hflip(self,img,mask=None,landmark=None,bbox=None):
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
	
	# def collate_fn(self,batch):
	# 	img_f,img_r=zip(*batch)
	# 	data={}
	# 	data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0).float()
	# 	data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f)).float()
	# 	return data

	def collate_fn(self,batch):
		videos,labels=zip(*batch)
		data={}
		data['video']=torch.tensor(videos)
		data['label']=torch.tensor(labels)
		return data
		
	def worker_init_fn(self,worker_id):                                                          
		np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__=='__main__':
	import blend as B
	from initialize import *
	from funcs import IoUfrom2bboxes,crop_face,RandomDownScale
	if exist_bi:
		from library.bi_online_generation import random_get_hull
	seed=10
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	image_dataset=Taylor_Dataset(phase='test',image_size=256)
	batch_size=64
	dataloader = torch.utils.data.DataLoader(image_dataset,
					batch_size=batch_size,
					shuffle=True,
					collate_fn=image_dataset.collate_fn,
					num_workers=0,
					worker_init_fn=image_dataset.worker_init_fn
					)
	data_iter=iter(dataloader)
	data=next(data_iter)
	img=data['img']
	img=img.view((-1,3,256,256))
	utils.save_image(img, 'loader.png', nrow=batch_size, normalize=False, range=(0, 1))
else:
	from sbi_utils import blend as B
	from .initialize import *
	from .funcs import IoUfrom2bboxes,crop_face,RandomDownScale
	if exist_bi:
		from sbi_utils.library.bi_online_generation import random_get_hull


