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
from multiprocessing import Process, Pool
np.set_printoptions(threshold=sys.maxsize)
import utils_caoc.kendall_tau_correlation as kendall_tau_correlation
import utils_caoc.order_functions as order_functions
import causal_viz.occlude_images_model as occlude_images_model
import causal_viz.images_by_patches as images_by_patches
import concurrent.futures


BATCH_SIZE = 64
WORKERS = 4
NUM_THREADS = 4

##############################################
#                Ms-IV                       #
##############################################

def multi_scale_viz(net, param,rank_original,probs_original, num_images, classes, level0, level_final, idx_img, size_tile,save_path,thr, image=None):

	
	trans = transforms.Compose([transforms.ToTensor()]) 


	num_final_side_tiles = level0 // size_tile                             #num of smallest size tile per image side                          
	number_tiles_top = num_final_side_tiles * num_final_side_tiles         #total number of smallest size tiles 
	leveln = int(np.sqrt(level0 // level_final))                           #last level in the hierarchy

	matrix_weights = np.zeros((num_final_side_tiles, num_final_side_tiles))


	#--function to split tile in coord vector into four other tiles and 
	#--calculate importance
	def process_coordinate(coord):

		matrix_weights_aux = np.zeros((num_final_side_tiles, num_final_side_tiles))
		matrix_local_values = np.zeros((num_tile_level, num_tile_level))

		for i in range(coord[0] * 2, coord[0] * 2 + 2):
			for j in range(coord[1] * 2, coord[1] * 2 + 2):

				base_x, base_y, num_units = images_by_patches.convert_coordinates_by_level(level0, num_final_side_tiles, level, i, j)	#relative coordinates to global coordinates			

				
				#--changing activation for one tile occluded image
				train_dataset_occlusion = occlude_images_model.MyDataset_tile_occlude_one_image(param, size_tile, number_tiles_top, num_images, [base_x, base_y], num_units, idx_img, transform=trans, image_modify = image)
				train_dataloader_occlusion = torch.utils.data.DataLoader(train_dataset_occlusion, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
				
				labels_aux = np.array([param.iloc[i]['labels'] for i in range(0, len(param), number_tiles_top)])
				probs_modified = probs_original.copy()
				prob_idx = occlude_images_model.extract_probs(net, train_dataloader_occlusion,idx_img)
				probs_modified[idx_img,:] = prob_idx[0,:]  
				#--


				#--CaOC
				feat_map_probs_occlusion = order_functions.create_order(probs_modified, labels_aux, classes)      #create new order
				new_corr, signal = kendall_tau_correlation.count_changes(rank_original[1, :], feat_map_probs_occlusion[1, :], idx_img)
				#--


				matrix_weights_aux = images_by_patches.sum_values_matrix(matrix_weights_aux, base_x, base_y, num_units, new_corr) #update importances in the aux matrix of tiles
				matrix_local_values[i, j] += new_corr #accumulate sum

		return matrix_weights_aux, matrix_local_values

	#--
	#--



	selected_patches = np.array([[0, 0]])   #we start with the complete image as a tile

	with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:     #parallelizing importance calculus
		for level in range(1, leveln + 1):
			
			num_tile_level = (2 ** level)

			coord_data = []
			for coord in selected_patches:
				coord_data.append(coord)

			results = executor.map(process_coordinate, coord_data)
			
			for result in results:                                             #combining results to obtain final hierarchical importance matrix
				
				matrix_weights_aux, matrix_local_values = result
				matrix_weights += images_by_patches.norm_matrix_max(matrix_weights_aux)   #norm by max importance value of the level
			selected_patches, threshold_value = images_by_patches.chose_next(level, matrix_local_values, thr)   #chose patches for next level according to threshold

	matrix_weights = images_by_patches.norm_matrix_max(matrix_weights)
	
	base_path = 'viz_images/'+save_path+'_image_idx' + str(idx_img) + '.jpg'
	occlude_images_model.save_map_as_images(matrix_weights, 1, 1, param, base_path, size_tile, None, number_tiles_top, None, idx_img=idx_img) #save visualization
	return matrix_weights

##############################################
#                Ms-IV                       #
##############################################



	
