from glob import glob
import os
import sys
import json
import numpy as np
from PIL import Image
from glob import glob 
import os
import pandas as pd
import random

def init_ff8(dataset='all',phase='train',level='frame',n_frames=8):
	dataset_path='data/FaceForensics-taylor/original_sequences/youtube/raw/frames/'

	image_list=[]
	label_list=[]

	folder_list = sorted(glob(dataset_path+'*'))
	filelist = []
	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	print('load from ', f'data/FaceForensics++/{phase}.json')
	for i in list_dict:
		filelist+=i
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	if level =='video':
		label_list=[0]*len(folder_list)
		return folder_list,label_list
	for i in range(len(folder_list)):
		# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		if n_frames<len(images_temp):
			if phase == 'train':
				images_temp = random.sample(images_temp, n_frames)
			else:
				images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
		image_list+=images_temp
		label_list+=[0]*len(images_temp)

	if phase == 'train':
		# import pdb; pdb.set_trace()
		if dataset=='all':
			fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
		else:
			fakes=[dataset]

		folder_list=[]
		for fake in fakes:
			fake_path=f'data/FaceForensics-taylor/manipulated_sequences/{fake}/raw/frames/'
			fake_folder_list = sorted(glob(fake_path+'*'))
			for i in range(len(fake_folder_list))[:900]:
				# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
				images_temp=sorted(glob(fake_folder_list[i]+'/*.png'))
				if n_frames<len(images_temp):
					images_temp = random.sample(images_temp, n_frames)
				image_list+=images_temp
				label_list+=[1]*len(images_temp)
	else:
		if dataset=='all':
			fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
		else:
			fakes=[dataset]

		folder_list=[]
		for fake in fakes:
			fake_path=f'data/FaceForensics-taylor/manipulated_sequences/{fake}/raw/frames/'
			fake_folder_list = sorted(glob(fake_path+'*'))
			for i in range(len(fake_folder_list))[900:]:
				# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
				images_temp=sorted(glob(fake_folder_list[i]+'/*.png'))
				if n_frames<len(images_temp):
					images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
				image_list+=images_temp
				label_list+=[1]*len(images_temp)
			
	return image_list,label_list


def init_rgb_ff8(dataset='all',phase='train',level='frame',n_frames=8):
	dataset_path='data/FaceForensics++/original_sequences/youtube/raw/frames/'

	image_list=[]
	label_list=[]

	folder_list = sorted(glob(dataset_path+'*'))
	filelist = []
	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	print('load from ', f'data/FaceForensics++/{phase}.json')
	for i in list_dict:
		filelist+=i
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	if level =='video':
		label_list=[0]*len(folder_list)
		return folder_list,label_list
	for i in range(len(folder_list)):
		# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		if n_frames<len(images_temp):
			if phase == 'train':
				images_temp = random.sample(images_temp, n_frames)
			else:
				images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
		image_list+=images_temp
		label_list+=[0]*len(images_temp)

	if phase == 'train':
		# import pdb; pdb.set_trace()
		if dataset=='all':
			fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
		else:
			fakes=[dataset]

		folder_list=[]
		for fake in fakes:
			fake_path=f'data/FaceForensics++/manipulated_sequences/{fake}/raw/frames/'
			fake_folder_list = sorted(glob(fake_path+'*'))
			for i in range(len(fake_folder_list))[:900]:
				# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
				images_temp=sorted(glob(fake_folder_list[i]+'/*.png'))
				if n_frames<len(images_temp):
					images_temp = random.sample(images_temp, n_frames)
				image_list+=images_temp
				label_list+=[1]*len(images_temp)
	else:
		if dataset=='all':
			fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
		else:
			fakes=[dataset]

		folder_list=[]
		for fake in fakes:
			fake_path=f'data/FaceForensics++/manipulated_sequences/{fake}/raw/frames/'
			fake_folder_list = sorted(glob(fake_path+'*'))
			for i in range(len(fake_folder_list))[900:]:
				# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
				images_temp=sorted(glob(fake_folder_list[i]+'/*.png'))
				if n_frames<len(images_temp):
					images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
				image_list+=images_temp
				label_list+=[1]*len(images_temp)
			
			# images_temp=sorted(glob(fake_path+'/*.png'))
			# if n_frames<len(images_temp):
			# 	images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
			# image_list+=images_temp
			# label_list+=[1]*len(images_temp)
	return image_list,label_list


def init_rgb_ff6(dataset='all',phase='train',level='frame',n_frames=8):
	dataset_path='data/FaceForensics++/original_sequences/youtube/raw/frames/'

	image_list=[]
	label_list=[]

	folder_list = sorted(glob(dataset_path+'*'))
	filelist = []
	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	print('load from ', f'data/FaceForensics++/{phase}.json')
	for i in list_dict:
		filelist+=i
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	if level =='video':
		label_list=[0]*len(folder_list)
		return folder_list,label_list
	for i in range(len(folder_list)):
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		if n_frames<len(images_temp):
			images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
		image_list+=images_temp
		label_list+=[0]*len(images_temp)

	if phase == 'train':
		if dataset=='all':
			fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
		else:
			fakes=[dataset]

		folder_list=[]
		for fake in fakes:
			fake_path=f'data/FaceForensics++/manipulated_sequences/{fake}/raw/frames/'
			fake_folder_list = sorted(glob(fake_path+'*'))
			for i in range(len(fake_folder_list))[:900]:
				# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
				images_temp=sorted(glob(fake_folder_list[i]+'/*.png'))
				if n_frames<len(images_temp):
					images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
				image_list+=images_temp
				label_list+=[1]*len(images_temp)
	else:
		if dataset=='all':
			fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
		else:
			fakes=[dataset]

		folder_list=[]
		for fake in fakes:
			fake_path=f'data/FaceForensics++/manipulated_sequences/{fake}/raw/frames/'
			fake_folder_list = sorted(glob(fake_path+'*'))
			for i in range(len(fake_folder_list))[900:]:
				images_temp=sorted(glob(fake_folder_list[i]+'/*.png'))
				if n_frames<len(images_temp):
					images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
				image_list+=images_temp
				label_list+=[1]*len(images_temp)
			
	return image_list,label_list


def init_ff6(dataset='all',phase='train',level='frame',n_frames=8):
	dataset_path='data/FaceForensics-taylor/original_sequences/youtube/raw/frames/'

	image_list=[]
	label_list=[]

	folder_list = sorted(glob(dataset_path+'*'))
	filelist = []
	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	print('load from ', f'data/FaceForensics++/{phase}.json')
	for i in list_dict:
		filelist+=i
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	if level =='video':
		label_list=[0]*len(folder_list)
		return folder_list,label_list
	for i in range(len(folder_list)):
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		if n_frames<len(images_temp):
			images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
		image_list+=images_temp
		label_list+=[0]*len(images_temp)

	if phase == 'train':
		if dataset=='all':
			fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
		else:
			fakes=[dataset]

		folder_list=[]
		for fake in fakes:
			fake_path=f'data/FaceForensics-taylor/manipulated_sequences/{fake}/raw/frames/'
			fake_folder_list = sorted(glob(fake_path+'*'))
			for i in range(len(fake_folder_list))[:900]:
				images_temp=sorted(glob(fake_folder_list[i]+'/*.png'))
				if n_frames<len(images_temp):
					images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
				image_list+=images_temp
				label_list+=[1]*len(images_temp)
	else:
		if dataset=='all':
			fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
		else:
			fakes=[dataset]

		folder_list=[]
		for fake in fakes:
			fake_path=f'data/FaceForensics-taylor/manipulated_sequences/{fake}/raw/frames/'
			fake_folder_list = sorted(glob(fake_path+'*'))
			for i in range(len(fake_folder_list))[900:]:
				# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
				images_temp=sorted(glob(fake_folder_list[i]+'/*.png'))
				if n_frames<len(images_temp):
					images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
				image_list+=images_temp
				label_list+=[1]*len(images_temp)
			
	return image_list,label_list


def init_ff(phase,level='frame',n_frames=8):
	dataset_path='data/FaceForensics++/original_sequences/youtube/raw/frames/'

	image_list=[]
	label_list=[]

	folder_list = sorted(glob(dataset_path+'*'))
	filelist = []
	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	print('load from ', f'data/FaceForensics++/{phase}.json')
	for i in list_dict:
		filelist+=i
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	if level =='video':
		label_list=[0]*len(folder_list)
		return folder_list,label_list
	for i in range(len(folder_list)):
		# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		if n_frames<len(images_temp):
			images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
		image_list+=images_temp
		label_list+=[0]*len(images_temp)

	return image_list,label_list


def init_ff_taylor(phase,level='frame',n_frames=8):
	dataset_path='data/FaceForensics-taylor/original_sequences/youtube/raw/frames/'

	image_list=[]
	label_list=[]

	folder_list = sorted(glob(dataset_path+'*'))
	filelist = []
	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	print('load from ', f'data/FaceForensics++/{phase}.json')
	for i in list_dict:
		filelist+=i
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	if level =='video':
		label_list=[0]*len(folder_list)
		return folder_list,label_list
	for i in range(len(folder_list)):
		# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		if n_frames<len(images_temp):
			images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
		image_list+=images_temp
		label_list+=[0]*len(images_temp)

	return image_list,label_list


def init_ff_video_taylor(dataset='all',phase='train'):
	assert dataset in ['all','Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	original_path='data/FaceForensics-taylor/original_sequences/youtube/raw/videos/'
	folder_list = sorted(glob(original_path+'*'))

	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	filelist=[]
	for i in list_dict:
		filelist+=i
	image_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
	label_list=[0]*len(image_list)

	if dataset=='all':
		fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	else:
		fakes=[dataset]

	folder_list=[]
	for fake in fakes:
		fake_path=f'data/FaceForensics-taylor/manipulated_sequences/{fake}/raw/videos/'
		folder_list_all=sorted(glob(fake_path+'*'))
		folder_list+=[i for i in folder_list_all if os.path.basename(i)[:3] in filelist]
	label_list+=[1]*len(folder_list)
	image_list+=folder_list
	return image_list,label_list


def init_ff_video(dataset='all',phase='train'):
	assert dataset in ['all','Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	original_path='data/FaceForensics++/original_sequences/youtube/raw/videos/'
	folder_list = sorted(glob(original_path+'*'))

	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	filelist=[]
	for i in list_dict:
		filelist+=i
	image_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
	label_list=[0]*len(image_list)

	if dataset=='all':
		fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	else:
		fakes=[dataset]

	folder_list=[]
	for fake in fakes:
		fake_path=f'data/FaceForensics++/manipulated_sequences/{fake}/raw/videos/'
		folder_list_all=sorted(glob(fake_path+'*'))
		folder_list+=[i for i in folder_list_all if os.path.basename(i)[:3] in filelist]
	label_list+=[1]*len(folder_list)
	image_list+=folder_list
	return image_list,label_list


def init_ff9(dataset='all',phase='train',level='frame',n_frames=8):
	dataset_path='data/FaceForensics++/original_sequences/youtube/raw/frames/'

	image_list=[]
	label_list=[]

	folder_list = sorted(glob(dataset_path+'*'))
	filelist = []
	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	print('load from ', f'data/FaceForensics++/{phase}.json')
	name_dict = {}
	for i in list_dict:
		filelist+=i
		name_dict[i[1]] = i[0]
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	if level =='video':
		label_list=[0]*len(folder_list)
		return folder_list,label_list
	for i in range(len(folder_list)):
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		if n_frames<len(images_temp):
			images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
		image_list+=images_temp
		label_list+=[0]*len(images_temp)
	return image_list,label_list,name_dict


def init_ff12(dataset='all',phase='train',level='frame',n_frames=8):
	dataset_path='data/FaceForensics-taylor/original_sequences/youtube/raw/frames/'

	image_list=[]
	label_list=[]

	folder_list = sorted(glob(dataset_path+'*'))
	filelist = []
	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	print('load from ', f'data/FaceForensics++/{phase}.json')
	name_dict = {}
	for i in list_dict:
		filelist+=i
		name_dict[i[1]] = i[0]
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	if level =='video':
		label_list=[0]*len(folder_list)
		return folder_list,label_list
	for i in range(len(folder_list)):
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		if n_frames<len(images_temp):
			images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
		image_list+=images_temp
		label_list+=[0]*len(images_temp)
	return image_list,label_list,name_dict


def init_ff6_continue(dataset='all',phase='train',level='frame',n_frames=8):
	dataset_path='data/FaceForensics-taylor/original_sequences/youtube/raw/frames/'

	image_list=[]
	label_list=[]

	folder_list = sorted(glob(dataset_path+'*'))
	filelist = []
	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	print('load from ', f'data/FaceForensics++/{phase}.json')
	for i in list_dict:
		filelist+=i
	folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

	if level =='video':
		label_list=[0]*len(folder_list)
		return folder_list,label_list
	for i in range(len(folder_list)):
		images_temp=sorted(glob(folder_list[i]+'/*.png'))
		if n_frames<len(images_temp):
			start = random.randint(0,len(images_temp)-n_frames)
			images_temp=[images_temp[round(i)] for i in np.linspace(start,start+n_frames-1,n_frames)]
		image_list+=images_temp
		label_list+=[0]*len(images_temp)

	if phase == 'train':
		if dataset=='all':
			fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
		else:
			fakes=[dataset]

		folder_list=[]
		for fake in fakes:
			fake_path=f'data/FaceForensics-taylor/manipulated_sequences/{fake}/raw/frames/'
			fake_folder_list = sorted(glob(fake_path+'*'))
			for i in range(len(fake_folder_list))[:900]:
				images_temp=sorted(glob(fake_folder_list[i]+'/*.png'))
				if n_frames<len(images_temp):
					start = random.randint(0,len(images_temp)-n_frames)
					images_temp=[images_temp[round(i)] for i in np.linspace(start,start+n_frames-1,n_frames)]
				image_list+=images_temp
				label_list+=[1]*len(images_temp)
	else:
		if dataset=='all':
			fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
		else:
			fakes=[dataset]

		folder_list=[]
		for fake in fakes:
			fake_path=f'data/FaceForensics-taylor/manipulated_sequences/{fake}/raw/frames/'
			fake_folder_list = sorted(glob(fake_path+'*'))
			for i in range(len(fake_folder_list))[900:]:
				# images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
				images_temp=sorted(glob(fake_folder_list[i]+'/*.png'))
				if n_frames<len(images_temp):
					images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
				image_list+=images_temp
				label_list+=[1]*len(images_temp)
			
	return image_list,label_list
