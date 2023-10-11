#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from kneed import KneeLocator
import utils_model.crop_feat_maps as crop_feat_maps
import utils_model.test_individual_feat_map as test_individual_feat_map
import causal_viz.representation_maps as representation_maps
import utils_caoc.order_functions as order_functions
from load_models import load_models
import os
import copy
from argparse import ArgumentParser


def save_top_image_clusters(net, groups, number_feat_maps, analyzed_layer,classes, label, top_images,train_dataloader, save_path):
	all_activation_probs = []
	for cluster in range(groups.max()+1):
		#--cluster definition part
		labels_bool = np.array(groups == np.full(groups.shape, cluster), dtype=bool)
		c_feat_maps = np.arange(len(labels_bool))[labels_bool]
		#--

		modified_model = test_individual_feat_map.zera(c_feat_maps,number_feat_maps,analyzed_layer, copy.deepcopy(net))
		print('Model modified!!!')

		all_activation_probs.append(order_functions.create_order(test_individual_feat_map.extract_probs(modified_model, train_dataloader,device=device),label, classes))
		print('Obtained orders from modified model!!!')
		
		idxs = list(np.argsort(all_activation_probs[-1][1,:])[:top_images])

		np.save(save_path+'idx_images_cluster'+str(cluster)+'.npy', np.array(idxs))




def find_clusters_params(net, train_dataloader, train_dataloader_tile,analyzed_layer, num_images, size_tile, number_tiles_top_representation,num_clusters,save_path):


	silhouette_complete = []
	distortion_complete = []
	inertia_complete = []


	file_path_save_representation = save_path+str(num_images)+'_representation_tile_size'+str(size_tile)+'_top'+str(number_tiles_top_representation)+'.npy'
	file_path_save_activations = save_path+str(num_images)+'_activations_tile_size'+str(size_tile)+'_top'+str(number_tiles_top_representation)+'.npy'
	file_path_save_params =save_path+ str(num_images)+'_params_tile_size'+str(size_tile)+'_top'+str(number_tiles_top_representation)+'.csv'

	occlusion_per_groups = []


	################################################################################################################################
	#load feature maps representation based on activations
	if not os.path.isfile(file_path_save_representation):
		print('Generating representation')
		representation = representation_maps.representation_tile_based(net, train_dataloader_tile, analyzed_layer,number_feat_maps, number_tiles_top_representation,size_tile,device=device)
		np.save(file_path_save_representation, representation)

	else:
		print('Loading representation')
		representation = np.load(file_path_save_representation)

	if not os.path.isfile(file_path_save_activations):
		print('Generating activations and parameters')
		activations, param = crop_feat_maps.obtain_activations(net, train_dataloader_tile,analyzed_layer,eval('net.'+analyzed_layer),device=device)
		np.save(file_path_save_activations, activations)
		param.to_csv(file_path_save_params)

	else:
		print('Loading activations and parameters')
		activations = np.load(file_path_save_activations)
		param = pd.read_csv(file_path_save_params)
		

	
	################################################################################################################################
	#cluster feature maps
	evaluation_clusters = []
	
	for cluster_n in range(2,num_clusters):

		groups, silhouette, distortion,inertia,_,_,_ = representation_maps.cluster_feat_maps(representation,save_path+str(cluster_n)+'cluster_scatter_num_images'+str(num_images)+'_size_tile'+str(size_tile)+'.png', num_clusters=cluster_n)


		evaluation_clusters.append([silhouette, distortion,inertia])


	evaluation_clusters = np.array(evaluation_clusters)
	

	np.save(save_path+'num_images'+str(num_images)+'_size_tile'+str(size_tile)+'_top'+str(number_tiles_top_representation)+'_kmeans_evaluation.npy', evaluation_clusters)

	
	
	kl = KneeLocator(range(2,num_clusters), evaluation_clusters[:,1], curve="convex", direction="decreasing")
	kl.plot_knee()

	print('The knee is:',kl.knee)
	
	
	################################################################################################################################
	#chosed k
	
	cluster_n = kl.knee

	groups, silhouette, distortion,inertia,pos_viz,_,_ = representation_maps.cluster_feat_maps(representation,save_path+str(cluster_n)+'chosen_clusters_num_images'+str(num_images)+'_size_tile'+str(size_tile)+'.png', num_clusters=cluster_n)


	np.save(save_path+'.npy', groups)

	
	return groups

################################################################################################################################

if __name__ == "__main__":


	parser = ArgumentParser()

	parser.add_argument("--model", type=str, default="resnet")
	parser.add_argument("--dataset", type=str, default="CUB")
	parser.add_argument("--min_patch", type=int, default=4)
	parser.add_argument("--patch_rep", type=int, default=5)
	parser.add_argument("--num_images", type=int, default=512)
	parser.add_argument("--top_img_ex", type=int, default=100)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--gpu_id", type=str, default="0")

	parser.set_defaults(feature=False)
	args = parser.parse_args()

	BATCH_SIZE = args.batch_size
	WORKERS = args.num_workers
	name_net = args.model
	dataset_name = args.dataset
	size_tile = args.min_patch
	number_tiles_top_representation = args.patch_rep
	classes = 2
	num_images = args.num_images
	top_images = args.top_img_ex

	if torch.cuda.is_available():  
	    device = "cuda:"+str(args.gpu_id) 
	else:  
	    device = "cpu"  

	torch.cuda.set_device(device)
	torch.multiprocessing.set_sharing_strategy('file_system')

	
	################################################################################################################################
	#clustering feature maps ###################
	np.random.seed(seed=0)
	torch.manual_seed(0)
	number_feat_maps = 512 #number of feature maps in last conv layer in VGG and Resnet
	num_clusters = 25 #max clusters to test
	save_path = 'save_clusters/clusters_'+name_net+'_'+dataset_name
	

	#loading model ###################
	net, path_label_file,path_images,name_label_file, analyzed_layer = load_models(name_net, dataset_name,classes)
	#path_label_file = '../../data_CUB/CUB_200_2011/'
	#path_images = '../../data_CUB/CUB_200_2011/images/'
	#name_label_file = 'intercalate_images_train.csv'
	net = net.to(device)
	net.eval()

	### DATALOADER ###############################################################################################
	trans = transforms.Compose([transforms.ToTensor()])

	final_to_intercalate = pd.read_csv(path_label_file+name_label_file)

	if dataset_name == 'CUB':
		train_dataset = test_individual_feat_map.MyDataset_CUB(path_label_file,path_images,name_label_file,num_images,transform=trans)
		train_dataset_tile = crop_feat_maps.MyDataset_tile_CUB(path_label_file,path_images,name_label_file,size_tile,num_images,transform=trans)


		class1 = np.arange(113,134) #sparrow (20 types)
		class2 = np.arange(158,179) #warbler until class 182 (20 types)
		final_to_intercalate = final_to_intercalate[np.logical_or(final_to_intercalate['class'].isin(class1),final_to_intercalate['class'].isin(class2))].reset_index(drop=True)
		final_to_intercalate['class'][final_to_intercalate['class'].isin(class1)] =  0
		final_to_intercalate['class'][final_to_intercalate['class'].isin(class2)] =  1

	else:
		train_dataset = test_individual_feat_map.MyDataset(path_label_file,path_images,name_label_file,num_images,transform=trans)
		train_dataset_tile = crop_feat_maps.MyDataset_tile(path_label_file,path_images,name_label_file,size_tile,num_images,transform=trans)


	train_dataloader = torch.utils.data.DataLoader(train_dataset,  batch_size=BATCH_SIZE,  shuffle=False,num_workers=WORKERS)
	train_dataloader_tile = torch.utils.data.DataLoader(train_dataset_tile,  batch_size=BATCH_SIZE,  shuffle=False,num_workers=WORKERS)


	#find clusters ###################

	groups = find_clusters_params(net, train_dataloader, train_dataloader_tile,analyzed_layer, num_images, size_tile, number_tiles_top_representation,num_clusters,save_path)

	
	label = 1-np.array(final_to_intercalate['class'].tolist())[:num_images] #cat 0 / dog 1



	save_top_image_clusters(net, groups, number_feat_maps, analyzed_layer,classes, label, top_images,train_dataloader, save_path)

	
