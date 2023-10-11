#!/usr/bin/env python
# coding: utf-8



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from argparse import ArgumentParser


########################################################
##                   DATASET CUB                      ##	
########################################################


class MyDataset_CUB(Dataset):
	
	def __init__(self,base,base_image,input_path, num_images = None, transform=None):

		self.class1 = np.arange(113,134) #sparrow
		self.class2 = np.arange(158,179) #warbler until class 182
		
		self.input_samples = pd.read_csv(base+input_path)
		
		self.input_samples = self.input_samples[np.logical_or(self.input_samples['class'].isin(self.class1),self.input_samples['class'].isin(self.class2))]
		self.input_samples = self.input_samples.loc[self.input_samples['class'] != -1].reset_index()

		
		self.transform = transform
		self.base = base
		self.base_image = base_image
		self.num_images = num_images
		

	def __getitem__(self, idx):
		
		image_path = self.input_samples['path'][idx]
		label = self.input_samples['class'][idx]

		sample =Image.open(self.base_image+image_path, mode='r').convert("RGB")	
		sample = sample.resize((224,224))

		
		train_mean = np.array([[[106.20628246,115.9285463,124.40483277]]], dtype=np.float32)
		train_std = np.array([[[65.59749505,64.94964833,66.61112731]]], dtype=np.float32)

		# zero mean and unit variance
		sample = (sample - train_mean) / train_std
		
		label = int(label)
		if label in self.class1:
			
			label =  [1,0]
			#print('class1')
		else:
			
			label =  [0,1]  
			#print('class2')


		if self.transform:
			sample = self.transform(sample)

		return torch.Tensor(sample), self.base_image+image_path,np.asarray(label).astype(np.float32).reshape(-1,)	

	def __len__(self):
		return len(self.input_samples)





########################################################
##               DATASET CAT_DOG                      ##	
########################################################

class MyDataset(Dataset):
	def __init__(self,base,base_image,input_path, transform=None):
		
		
		self.input_samples = pd.read_csv(base+input_path)
		self.input_samples = self.input_samples.loc[self.input_samples['label'] != -1].reset_index()
		self.transform = transform
		self.base = base
		self.base_image = base_image
		

	def __getitem__(self, idx):
		
		image_path = self.input_samples['path'][idx]
		label = self.input_samples['label'][idx]
		sample =Image.open(self.base_image+image_path, mode='r').convert("RGB")	
		
		sample = sample.resize((224,224))

		
		train_mean = np.array([[[106.20628246,115.9285463,124.40483277]]], dtype=np.float32)
		train_std = np.array([[[65.59749505,64.94964833,66.61112731]]], dtype=np.float32)
		# zero mean and unit variance
		sample = (sample - train_mean) / train_std

		
		label = int(label)
		if label==1:
			
			label =  [1,0]
		else:
			
			label =  [0,1]

		

		if self.transform:
			sample = self.transform(sample)

		

		return torch.Tensor(sample), self.base_image+image_path,np.asarray(label).astype(np.float32).reshape(-1,)	

	def __len__(self):
		return len(self.input_samples)



########################################################
##               TEST FUNCTION                       ##	
########################################################

def test(name_save,net, dataloader, val_loss=-1,cont=0):
	correct = 0
	total = 0
	correct_0 = 0
	label_0 = 0
	label_1 = 0
	correct_1 = 0
	loss = 0
	prob = nn.Softmax(dim = 1)
	net.eval()
	with torch.no_grad():
		for i,data in enumerate(dataloader):
			print('Batch:', i)
			images,_, labels = data
			images, labels = images.to(device), labels.to(device)

			outputs = prob(net(images))
			loss +=  torch.nn.functional.binary_cross_entropy(outputs, labels)
			_, predicted = torch.max(outputs.data, 1)
			_, labels_aux = torch.max(labels, 1)
			total += labels.size(0)
			correct += (predicted == labels_aux).sum().item()
			label_1 += labels_aux.sum()
			label_0 += len(labels_aux) - labels_aux.sum()
			for t in range(len(predicted)):
			
				correct_1 += ((predicted[t] == labels_aux[t]) and (labels_aux[t] == 1)).sum().item()
				correct_0 += ((predicted[t] == labels_aux[t]) and (labels_aux[t] == 0)).sum().item()
	
	##saving only best	
	if loss< val_loss:
		val_loss = loss
		torch.save(net.state_dict(), name_save+'weights_only.pth')
		cont=0
	else:
		cont+=1
	
	print('Loss val:', loss/i)
	print('Accrracy: %f %%' % (100 * correct / total))
	print('Class 0: %f %%' % (100 * correct_0 / label_0))
	print('Class 1: %f %%' % (100 * correct_1 / label_1))
	print(correct_0,correct_1, label_0, label_1)

	return val_loss,(100 * correct / total), cont

########################################################
##               TRAIN FUNCTION                       ##	
########################################################
def train(name_save,net, dataloader, val_dataloader, optimizer, criterion,nb_epochs,early_stop):
	val_loss = np.inf
	prob = nn.Softmax(dim = 1)
	cont=0
	net.train()
	for epoch in range(nb_epochs):
		print('Epoch:', epoch)
		running_loss = 0.0
		for i, data in enumerate(dataloader, 0):
			inputs,_, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			optimizer.zero_grad()
			outputs = prob(net(inputs))
			loss =  torch.nn.functional.binary_cross_entropy(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			
		print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / i))

		print('Val loss:', val_loss)
		val_loss,val_acc,cont = test(name_save,net, val_dataloader,val_loss,cont)
		if cont>=early_stop:
			break
		running_loss = 0.0
		net.train()
	print('Finished Training')
	return val_acc


######################################################################################################

if __name__ == "__main__":
	parser = ArgumentParser()

	
	parser.add_argument("--batch_size", type=int, default=256)
	parser.add_argument("--max_epochs", type=int, default=500)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--gpu_id", type=str, default="0")
	parser.add_argument("--learning_rate", type=float, default=1e-7)
	parser.add_argument("--early_stop", type=float, default=20)
	parser.add_argument("--weight_decay", type=float, default=1e-5)
	
	parser.add_argument("--train", type=int, default=1, choices=[0, 1])
	parser.add_argument("--model", type=str, default="vgg")
	parser.add_argument("--dataset", type=str, default="cat_dog")
	parser.add_argument("--classes", type=int, default=2)
	
	parser.set_defaults(feature=False)

	args = parser.parse_args()

	BATCH_SIZE = args.batch_size
	dataset = args.dataset
	model_name = args.model
	num_workers = args.num_workers
	gpu_id = args.gpu_id
	EPOCHS = args.max_epochs
	learning_rate = args.learning_rate
	weight_decay = args.weight_decay
	early_stop = args.early_stop
	train = args.train
	img_classes = args.classes


	device = torch.device('cuda:'+gpu_id if torch.cuda.is_available() else 'cpu')
	print(device)


	### DATALOADERS ###############################################################################################


	trans = transforms.Compose([transforms.ToTensor()])


	if dataset == 'cat_dog':
		train_dataset = MyDataset('datasets/'+dataset+'/','dataset/'+dataset+'/images/','train_paths.csv',transform=trans)
		val_dataset = MyDataset('datasets/'+dataset+'/','dataset/'+dataset+'/images/','val_paths.csv',transform=trans)
		test_dataset = MyDataset('datasets/'+dataset+'/','dataset/'+dataset+'/images/','train_paths.csv',transform=trans)
	elif dataset == 'CUB':
		train_dataset = MyDataset_CUB('datasets/'+dataset+'/','dataset/'+dataset+'/images/','intercalate_images_train.csv',transform=trans)
		val_dataset = MyDataset_CUB('datasets/'+dataset+'/','dataset/'+dataset+'/images/','intercalate_images_val.csv',transform=trans)
		test_dataset = MyDataset_CUB('datasets/'+dataset+'/','dataset/'+dataset+'/images/','intercalate_images_train.csv',transform=trans)
	else:
		print('Dataset not defined!')
		exit()


	
	train_dataloader = torch.utils.data.DataLoader(train_dataset, 
		                                       batch_size=BATCH_SIZE, 
		                                       shuffle=True,num_workers=num_workers)
	val_dataloader = torch.utils.data.DataLoader(val_dataset,
		                                     batch_size=BATCH_SIZE, 
		                                     shuffle=False,num_workers=num_workers)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, 
		                                      batch_size=BATCH_SIZE, 
		                                      shuffle=False,num_workers=num_workers)


	### MODEL ###############################################################################################

	acc_val = []

	np.random.seed(seed=0)
	torch.manual_seed(0)
	weights = 'checkpoints/'+dataset+'/'+model+'_weights.pth'
	if model_name == 'resnet':

		net = models.resnet18(pretrained=True)
		dim_in = net.fc.in_features
		net.fc = nn.Linear(dim_in, img_classes)

	elif  model_name == 'vgg':
		net = models.vgg16(pretrained=True)
		dim_in = net.classifier[6].in_features
		net.classifier[6] = nn.Linear(dim_in, img_classes)
	else:
		print('Model not defined!')
		exit()


	net.load_state_dict(torch.load(weights),strict=True)
	net = net.to(device)


	if train == 1:


		### TRAIN ###############################################################################################

		name_save = 'checkpoints/'+dataset+'/'+model
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
		acc = train(name_save,net, train_dataloader, val_dataloader, optimizer, criterion,EPOCHS,early_stop)

		print('Accuracy:', acc)

	else:
		### TEST ###############################################################################################
		net.eval()
		test('',net, val_dataloader)
	
