#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy
import sys
import os
import torchvision.transforms as transforms
np.set_printoptions(threshold=sys.maxsize)
import utils_model.crop_feat_maps as crop_feat_maps
import utils_caoc.order_functions as order_functions
import utils_model.test_individual_feat_map as test_individual_feat_map
import ms_iv
from load_models import load_models
from argparse import ArgumentParser
from PIL import Image


if __name__ == "__main__":


	parser = ArgumentParser()

	parser.add_argument("--model", type=str, default="resnet")
	parser.add_argument("--dataset", type=str, default="CUB")
	parser.add_argument("--min_patch", type=int, default=7)
	parser.add_argument("--max_patch", type=int, default=224)
	parser.add_argument("--patch_rep", type=int, default=5)
	parser.add_argument("--num_images", type=int, default=512)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--num_threads", type=int, default=4)
	parser.add_argument("--thr_viz", type=float, default=0.75)
	parser.add_argument("--gpu_id", type=str, default="0")

	parser.set_defaults(feature=False)
	args = parser.parse_args()

	BATCH_SIZE = args.batch_size
	WORKERS = args.num_workers
	NUM_THREADS = args.num_threads
	name_net = args.model
	dataset_name = args.dataset
	size_tile = args.min_patch
	level_final = args.min_patch
	level0 = args.max_patch
	number_tiles_top_representation = args.patch_rep
	classes = 2
	num_images = args.num_images
	thr = args.thr_viz
#!/usr/bin/env python
# coding: utf-8


	np.random.seed(seed=0)
	torch.manual_seed(0)


	if torch.cuda.is_available():  
	    device = "cuda:"+str(args.gpu_id) 
	else:  
	    device = "cpu"  

	torch.cuda.set_device(device)
	torch.multiprocessing.set_sharing_strategy('file_system')

	weights_model = ''
	path_label_file = ''
	path_images = ''
	name_label_file = ''

	

	number_feat_maps = 512
	
	#####################################################################3
	
	net, path_label_file,path_images,name_label_file,analyzed_layer= load_models(name_net, dataset_name,classes)	

	#path_label_file = '../../data_CUB/CUB_200_2011/'
	#path_images = '../../data_CUB/CUB_200_2011/images/'
	#name_label_file = 'intercalate_images_train.csv'


	net = net.to(device)
	net.eval()

	

	final_to_intercalate = pd.read_csv(path_label_file+name_label_file)
	trans = transforms.Compose([transforms.ToTensor()])

	#CUB
	if dataset_name == 'CUB':
		class1 = np.arange(113,134) #sparrow (20 types)
		class2 = np.arange(158,179) #warbler until class 182 (20 types)

		final_to_intercalate = final_to_intercalate[np.logical_or(final_to_intercalate['class'].isin(class1),final_to_intercalate['class'].isin(class2))].reset_index(drop=True)

		final_to_intercalate['class'][final_to_intercalate['class'].isin(class1)] =  0
		final_to_intercalate['class'][final_to_intercalate['class'].isin(class2)] =  1

		train_dataset = test_individual_feat_map.MyDataset_CUB(path_label_file,path_images,name_label_file,num_images,transform=trans)

	#cat_dog
	else:
		train_dataset = test_individual_feat_map.MyDataset(path_label_file,path_images,name_label_file,num_images,transform=trans)



	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=False,num_workers=WORKERS)


	
	load_path = 'save_clusters/clusters_'+name_net+'_'+dataset_name
	clusters = np.load('save_clusters/clusters_'+name_net+'_'+dataset_name+'.npy')  
	


	
	label = 1-np.array(final_to_intercalate['class'].tolist())[:num_images] 

	#masking network feature maps to visualize concepts
	for cluster in range(clusters.max()+1):

		print('Cluster:', cluster)

		labels_bool = np.array(clusters == np.full(clusters.shape, cluster), dtype=bool)#[:num_images_ranking]
		c_feat_maps = np.arange(len(labels_bool))[labels_bool]

		
		modified_model = test_individual_feat_map.zera(c_feat_maps,number_feat_maps,analyzed_layer, copy.deepcopy(net)) #zeroing out non-concept feature maps

		probs_original = test_individual_feat_map.extract_probs(modified_model, train_dataloader,device=device)         #get final activations for the images
		rank_original = order_functions.create_order(probs_original,label, classes)                                     #ranking according to activations




		#--getting patch's dataset 
		if dataset_name == 'CUB':
			train_dataset_tile = crop_feat_maps.MyDataset_tile_CUB(path_label_file,path_images,name_label_file,size_tile,num_images,transform=trans)
		else:
			train_dataset_tile = crop_feat_maps.MyDataset_tile(path_label_file,path_images,name_label_file,size_tile,num_images,transform=trans)
		train_dataloader_tile = torch.utils.data.DataLoader(train_dataset_tile,  batch_size=BATCH_SIZE, shuffle=False,num_workers=WORKERS)


		file_path_save_params = load_path+str(num_images)+'_params_tile_size'+str(size_tile)+'_top'+str(number_tiles_top_representation)+'.csv'
		if not os.path.isfile(file_path_save_params):
			print('Generating activations and parameters')
			activations, param = crop_feat_maps.obtain_activations(net, train_dataloader_tile,analyzed_layer,eval('net.'+analyzed_layer),device=device)
			param.to_csv(file_path_save_params)
		else:
			print('Loading activations and parameters')
			param = pd.read_csv(file_path_save_params)
		#--



		
		list_images = np.load('save_clusters/clusters_'+name_net+'_'+dataset_name+'idx_images_cluster'+str(cluster)+'.npy')



		for idx_img in list_images:
			matrix_weights = ms_iv.multi_scale_viz(modified_model,param,rank_original,probs_original,num_images,classes,level0,level_final,idx_img,size_tile, dataset_name+'/cluster'+str(cluster)+'_model_'+name_net,thr)

			matrix_weights[matrix_weights<(matrix_weights.max()*0.5)] = 0
			matrix_weights[matrix_weights>0] = 1

			matrix_weights = Image.fromarray(np.uint8(matrix_weights))
			matrix_weights = matrix_weights.resize((224,224))

			matrix_weights = np.array(matrix_weights)
			
			print(matrix_weights.shape, matrix_weights.max(),matrix_weights.min())
			np.save('viz_images/'+dataset_name+'/image'+str(idx_img)+'_cluster'+str(cluster)+'_model_'+name_net+'.npy',matrix_weights)
