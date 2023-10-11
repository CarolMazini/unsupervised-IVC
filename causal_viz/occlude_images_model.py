#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageEnhance
import copy
import torchvision.transforms as T


#--------------------------------------------------------------

def find_image_position(idx, num_tiles_side):

	num_tiles = num_tiles_side*num_tiles_side

	real_idx = idx//num_tiles

	tile_idx = idx % num_tiles

	tile_x = tile_idx // num_tiles_side
	
	tile_y =  tile_idx % num_tiles_side

	return real_idx, tile_x, tile_y




def crop_image(img, size_tile, tile_x, tile_y):

	box = (tile_y*size_tile, tile_x*size_tile, tile_y*size_tile+size_tile, tile_x*size_tile+size_tile)

	return img.crop(box)


	

class MyDataset_tile_occlude(Dataset):
	def __init__(self,params,size_tile,top,num_images=None, transform=None, occlude = True):
		
		
		self.params = params

		self.top = top

		self.filenames = params['filename']
		self.labels = params['labels']
		self.tile_x = params['tile_x']
		self.tile_y = params['tile_y']

		self.size_tile = size_tile

		self.num_tiles_side = (224// self.size_tile)
		
		self.transform = transform

		self.occlude = occlude
		
		self.num_images = num_images
		#print(self.filenames, self.labels)
		
		

	def __getitem__(self, idx):


		if self.occlude == True:


			idx_real = idx*self.top

			image_path = self.filenames.iloc[idx_real]
			sample =Image.open(image_path, mode='r').convert("RGB")	
			sample = sample.resize((224,224))
			label = self.labels.iloc[idx_real]

			for i in range(self.top):

				tile_x = self.tile_x.iloc[idx_real+i]
				tile_y = self.tile_y.iloc[idx_real+i]

				result = Image.new(sample.mode, (self.size_tile, self.size_tile), (0, 0, 0))
				sample.paste(result, (tile_y*self.size_tile, tile_x*self.size_tile))

		else:


			idx_real = idx

			image_path = self.filenames.iloc[idx_real]
			sample =Image.open(image_path, mode='r').convert("RGB")	
			sample = sample.resize((224,224))
			label = self.labels.iloc[idx_real]

			real_idx, tile_x, tile_y = find_image_position(idx, self.num_tiles_side)
			sample = crop_image(sample, self.size_tile, tile_x, tile_y)
			sample = sample.resize((224,224))
		
		#sample.save(str(idx)+'output_occlusion_final.jpg')
		
		train_mean = np.array([[[106.20628246,115.9285463,124.40483277]]], dtype=np.float32)
		train_std = np.array([[[65.59749505,64.94964833,66.61112731]]], dtype=np.float32)
		# zero mean and unit variance
		sample = (sample - train_mean) / train_std


		if self.transform:
			sample = self.transform(sample)

		

		return torch.Tensor(sample), image_path, np.asarray(label).astype(np.float32).reshape(-1,)	

	def __len__(self):

		if self.num_images == None:
			if self.occlude == True:
				size = len(self.params)//self.top
			else:
				size = len(self.params)
		else:
			if self.occlude == True:
				size = self.num_images
			else:
				size = self.num_images*self.top
		return size



class MyDataset_tile_occlude_one_image(Dataset):
	def __init__(self,params,size_tile,top,num_images,coord, quant,idx_img, transform=None, image_modify = None):
		
		
		self.params = params

		self.top = top
		self.coord = coord
		self.quant = quant
		self.idx_img = idx_img

		self.filenames = params['filename']
		self.labels = params['labels']
		self.tile_x = params['tile_x']
		self.tile_y = params['tile_y']

		self.size_tile = size_tile

		self.num_tiles_side = (224// self.size_tile)
		
		self.transform = transform

		
		self.num_images = num_images
		self.image_modify = image_modify
		

	def __getitem__(self, idx):

		idx_real = idx*self.top

		image_path = self.filenames.iloc[idx_real]
		sample =Image.open(image_path, mode='r').convert("RGB")	
		sample = sample.resize((224,224))
		label = self.labels.iloc[idx_real]

		train_mean = np.array([[[106.20628246,115.9285463,124.40483277]]], dtype=np.float32)
		train_std = np.array([[[65.59749505,64.94964833,66.61112731]]], dtype=np.float32)

		
		
		if idx == self.idx_img:

			if (self.image_modify) != None:
				sample = self.image_modify.permute(1,2,0).detach().cpu().numpy()*train_std +train_mean
				sample = Image.fromarray(np.uint8(sample))

			for i in range(self.coord[0], self.coord[0]+self.quant):
				for j in range(self.coord[1], self.coord[1]+self.quant):

					tile_x = self.tile_x.iloc[idx_real+(i*self.num_tiles_side)+j]
					tile_y = self.tile_y.iloc[idx_real+(i*self.num_tiles_side)+j]
					result = Image.new(sample.mode, (self.size_tile, self.size_tile), (0, 0, 0))
					sample.paste(result, (tile_y*self.size_tile, tile_x*self.size_tile))

		
		
		# zero mean and unit variance
		sample = (sample - train_mean) / train_std

		if self.transform:
			sample = self.transform(sample)
		

		return torch.Tensor(sample), image_path, np.asarray(label).astype(np.float32).reshape(-1,)	

	def __len__(self):

		if self.num_images == None:	
			size = len(self.params)//self.top
		else:	
		 	size = self.num_images	
		return size




def extract_probs(net, dataloader, idx=None):

	'''
	Input: model and dataloader with images to extract activations
	Output: matrix of extracted final probabilities (before softmax)
	'''

	activations = []
	
	net.eval()
	with torch.no_grad():

		if idx==None:
			for i,data in enumerate(dataloader):
				
				images,_, labels = data
				images, labels = images.to(device), labels.to(device)
				outputs = net(images)
				activations.extend(np.array(outputs.detach().cpu()))
		else:
			image,_, labels = dataloader.dataset[idx]
			outputs = net(torch.Tensor(image).unsqueeze(0).cuda())
			activations = np.array(outputs.detach().cpu())
			
	
	return np.array(activations)



def save_occlusion_as_images(max_cols, max_rows, data, path_save, size_tile, orders, top, filter_label):
		
	data['order'] = np.repeat(orders[1,:], top)
	data['label_single'] = np.repeat(orders[0,:], top)

	if filter_label != None:
		data = data[data['label_single'] == filter_label].reset_index()

	data_ordered = data.sort_values(by=['order','label_single'], ascending=True).reset_index()
	fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20,20))
	name_images = data_ordered['filename'].unique()

	

	for idx,path in enumerate(name_images):

		image =Image.open(path, mode='r').convert("RGB")	
		image = image.resize((224,224))
		filtered = data_ordered[data_ordered['filename']==path]
		result = copy.copy(image)
		enhancer = ImageEnhance.Contrast(result)
		result = enhancer.enhance(0.2)

		for line in range(len(filtered)):

			crop = crop_image(image, size_tile, filtered.iloc[line]['tile_x'],  filtered.iloc[line]['tile_y'])
			result.paste(crop, (filtered.iloc[line]['tile_y']*size_tile,  filtered.iloc[line]['tile_x']*size_tile))
		

		ax = fig.add_subplot(max_rows, max_cols, 1 + idx)

		ax.imshow(result, aspect="auto") #, cmap="gray"

		ax.xaxis.set_ticklabels([])
		ax.yaxis.set_ticklabels([]) 
		ax.tick_params(labelleft=False, length=0)
		ax.tick_params(labelbottom=False, length=0)
		if idx == (max_cols*max_rows-1):
			break


	plt.subplots_adjust(wspace=.05, hspace=.05)
	plt.savefig(path_save)


def save_map_as_images(matrix_weights, max_cols, max_rows, data, path_save, size_tile, orders, top, filter_label, idx_img = None):

	if idx_img == None:
		
		data['order'] = np.repeat(orders[1,:], top)
		data['label_single'] = np.repeat(orders[0,:], top)

		if filter_label != None:
			data = data[data['label_single'] == filter_label].reset_index()

		data_ordered = data.sort_values(by=['order','label_single'], ascending=True).reset_index()

		name_images = data_ordered['filename'].unique()
	else:

		name_images = [data.iloc[idx_img*top]['filename']]

	fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20,20))

	for idx,path in enumerate(name_images):

		image =Image.open(path, mode='r').convert("RGB")	
		image = image.resize((224,224))

		num_tiles = 224//size_tile

		for i in range(num_tiles):

			for j in range(num_tiles):

				crop = crop_image(image, size_tile, i, j)
				enhancer = ImageEnhance.Brightness(crop)
				crop = enhancer.enhance(matrix_weights[i,j])
				image.paste(crop, (j*size_tile,  i*size_tile))
				
		
		ax = fig.add_subplot(max_rows, max_cols, 1 + idx)

		ax.imshow(image, aspect="auto") #, cmap="gray"

		ax.xaxis.set_ticklabels([])
		ax.yaxis.set_ticklabels([]) 

		ax.tick_params(labelleft=False, length=0)

		ax.tick_params(labelbottom=False, length=0)

		if idx == (max_cols*max_rows-1):
			break


	plt.subplots_adjust(wspace=.05, hspace=.05)
	plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
	plt.savefig(path_save)




		
