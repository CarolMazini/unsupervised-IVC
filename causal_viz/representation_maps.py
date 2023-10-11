#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
import os
import umap
import utils_model.crop_feat_maps as crop_feat_maps
from sklearn.cluster import KMeans, SpectralClustering,AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.spatial.distance import cdist
from matplotlib.offsetbox import OffsetImage


def getImage(path, zoom=1):
    return OffsetImage(plt.imread(path), zoom=zoom)

def scatter_with_images(data,path_images, label_classe, subtitles, cores, xlabel, ylabel,save_path):


	plt.style.context('white')

	fig = plt.figure()
	fig.set_size_inches(6, 4)
	ax = SubplotZero(fig, 111)
	fig.add_subplot(ax)



	for direction in ["left", "bottom"]:
	    # adds arrows at the ends of each axis
	    ax.axis[direction].set_axisline_style("-|>")

	    # adds X and Y-axis from the origin
	    ax.axis[direction].set_visible(True)

	#for direction in ["left", "right", "bottom", "top"]:
	for direction in ["right", "top"]:
	    # hides borders
	    ax.axis[direction].set_visible(False)

	
	markers = ['o', 'd', '*', 's', '+']
	
	for i in range(len(subtitles)):

			mask = label_classe == i
			
			ax.scatter(data[mask,0],data[mask,1], marker=markers[0], color=cores[i], facecolors=cores[i],label = subtitles[i],s=5) #,label = subtitles[i]
	

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	ax.legend(loc='best')
	plt.grid(True)
	plt.savefig(save_path)


	
def scatter_plot_data(data, label_classe, subtitles, cores, xlabel, ylabel,save_path,annotation=None,black_points = None):

	plt.style.context('white')

	fig = plt.figure()
	fig.set_size_inches(6, 4)
	ax = SubplotZero(fig, 111)
	fig.add_subplot(ax)



	for direction in ["left", "bottom"]:
	    # adds arrows at the ends of each axis
	    ax.axis[direction].set_axisline_style("-|>")

	    # adds X and Y-axis from the origin
	    ax.axis[direction].set_visible(True)

	#for direction in ["left", "right", "bottom", "top"]:
	for direction in ["right", "top"]:
	    # hides borders
	    ax.axis[direction].set_visible(False)


	markers = ['o', 'd', '*', 's', '+']
	
	for i in range(len(subtitles)):

			mask = label_classe == i
			
			cor = np.full(len(label_classe[mask]),cores[i])

			ax.scatter(data[mask,0],data[mask,1], marker=markers[0], color=cores[i], facecolors='None',label = subtitles[i]) #,label = subtitles[i]
			

	if not annotation == None:
		for i, txt in enumerate(annotation):

			if i in black_points:
				ax.annotate(txt, (data[i,0], data[i,1]), color='k', weight='bold', fontsize=7)
			else:
    				ax.annotate(txt, (data[i,0], data[i,1]), color='grey', weight='bold', fontsize=6)
	
	
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	
	ax.legend(loc='best')
	

	plt.grid(True)

	plt.savefig(save_path)




def representation_tile_based(net, train_dataloader_tile, analyzed_layer,number_feat_maps, number_tiles_top,size_tile,device='cpu'):

	'''
	Input: network, tiles dataloader,analyzed layer, number of feature maps, number of tiles used to represent, size of each tile
	Output: the representation of the chosen feature maps 
	'''

	activations, param = crop_feat_maps.obtain_activations(net, train_dataloader_tile,analyzed_layer,eval('net.'+analyzed_layer),device=device)

	representation = []

	for feat_map in range(number_feat_maps):


		chosen_tiles, param_original = crop_feat_maps.find_best_per_feat(activations,param, feat_map, number_tiles_top, size_tile)

		representation.append(np.concatenate((param_original['tile_x'].to_numpy(), param_original['tile_y'].to_numpy())))

		
	return np.array(representation)


def cluster_feat_maps(representation,path_save, num_clusters=5):

	colors = ['#fff100', '#ff8c00',  '#e81123', '#ec008c', '#68217a','#00188f','#00bcf2','#00b294', '#009e49','#bad80a','#ff796c', '#580f41',  '#8c000f', '#ffc0cb', '#aaa662','#c79fef','#000000','#fc5a50', '#dbb40c','#a9561e']
	
	

	#viz_vector = umap.UMAP(random_state=0, n_neighbors = 100, min_dist = 0.0, metric = 'euclidean').fit_transform(representation)

	#viz_vector = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=80).fit_transform(representation)
	
	pca = PCA().fit(representation) #n_components=2
	sum_final = 0
	for i,s in enumerate(pca.explained_variance_ratio_):
		sum_final+=s
		if sum_final>=0.98: break

	print('Components:', i)
	print(np.sum(pca.explained_variance_ratio_[:i]))
	print(len(pca.explained_variance_ratio_))
	representation = pca.transform(representation)#[:,:i]
	
	cluster = KMeans(n_clusters=num_clusters, n_init=15,random_state=0,max_iter=1000).fit(representation)
	#cluster = AgglomerativeClustering(n_clusters=num_clusters).fit(representation)
	#cluster = SpectralClustering(n_clusters=num_clusters,assign_labels='kmeans',random_state=0).fit(representation)


	##to visualize
	viz_vector = umap.UMAP(random_state=0, n_neighbors = 50, min_dist = 0.0, metric = 'euclidean').fit_transform(representation)
	#scatter_plot_data(viz_vector, cluster.labels_, np.arange(num_clusters), colors[:num_clusters], 'x', 'y',path_save)

	silhouette = metrics.silhouette_score(representation, cluster.labels_, metric='euclidean')

	distortions = np.sum(np.min(cdist(representation, cluster.cluster_centers_,'euclidean'), axis=1)) / representation.shape[0]
	#distortions = None

	inertias = cluster.inertia_

	print(num_clusters,'clusters',silhouette, distortions, inertias)
	
	return cluster.labels_,silhouette, distortions, inertias,viz_vector,cluster.cluster_centers_,representation






