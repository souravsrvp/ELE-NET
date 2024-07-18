import torch
import torch.nn as nn
from dataset2 import ELE_Dataset
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm
import numpy as np
from ELE_Net import ELE_Net
from torch import tensor
from torch.utils.tensorboard import SummaryWriter
import os


def weight_histograms(writer, step, model): 
	print("Visualizing model weights...")
	'''
	This code can not handle CNNs yet
	'''
	for name, param in model.named_parameters():
		weight_bias = torch.flatten(param)
		writer.add_histogram(name, weight_bias, global_step=step, bins='tensorflow')


writer = SummaryWriter()

def save_model(model, model_path):
	torch.save(model.state_dict(), model_path)


def train(left_model, right_model, train_dataloader, epochs, 
		  optimizer_l, optimizer_r, criterion, output_classes, base_path, batch_size):
	

	for epoch in tqdm(range(1,epochs+1)):
		left_model.train()
		right_model.train()
		
		lypred = []
		lytrue = []
		rypred = []
		rytrue = []
		lacc = Accuracy(task="multiclass", num_classes=output_classes)
		racc = Accuracy(task="multiclass", num_classes=output_classes)
	
		current_loss_l = 0
		current_loss_r = 0
		loss_idx = 0
		for batch_ips in tqdm(train_dataloader):
			
			lpred = left_model(batch_ips[0])
			rpred = right_model(batch_ips[2])
			
			loss_l = criterion(lpred, batch_ips[1])
			loss_r = criterion(rpred, batch_ips[3])
			 
			loss_l = loss_l.mean()
			loss_r = loss_r.mean()
			
			#writer.add_scalar("Loss/MiniBatches", loss.item(), loss_idx+(epoch*len(train_dataset)/(batch_size)))
			
			loss_l.backward()
			loss_r.backward()
			loss_idx +=1
			current_loss_l = (current_loss_l*(loss_idx-1)+loss_l.detach().item())/loss_idx
			current_loss_r = (current_loss_r*(loss_idx-1)+loss_r.detach().item())/loss_idx
			optimizer_l.step()
			optimizer_l.zero_grad()
			
			optimizer_r.step()
			optimizer_r.zero_grad()
			#batch_loss.append(current_loss)
			lypred+=(torch.argmax(lpred, axis=1)).tolist()
			lytrue+=(batch_ips[1]).tolist()

			rypred+=(torch.argmax(rpred, axis=1)).tolist()
			rytrue+=(batch_ips[3]).tolist()
			#weight_histograms(writer, epoch, model)
			
		# weight_histograms(writer, epoch, left_model)
		# weight_histograms(writer, epoch, right_model)
		laccuracy = lacc(tensor(lypred), tensor(lytrue))
		raccuracy = racc(tensor(rypred), tensor(rytrue))

		writer.add_scalar("Left Loss/Epoch", current_loss_l, epoch)
		writer.add_scalar("Left Train Accuracy/Epoch", laccuracy.item()*100, epoch)

		writer.add_scalar("Right Loss/Epoch", current_loss_r, epoch)
		writer.add_scalar("Right Train Accuracy/Epoch", raccuracy.item()*100, epoch)

		print("Epoch:", epoch, "Left Loss:", current_loss_l, "Left Train Accuracy:", laccuracy.item()*100)
		print("Epoch:", epoch, "Right Loss:", current_loss_r, "Right Train Accuracy:", raccuracy.item()*100)

		# if epoch%2==0:
		# 	testacc = test(model=model, dataloader=test_dataloader, output_classes=output_classes)
		# 	model_path = base_path+'epoch_'+str(epoch)+'.pth'
		# 	if testacc > best_acc:
		# 		best_acc = testacc
		# 		save_model(model, model_path)
		model_path_l = base_path+'left'+'epoch_'+str(epoch)+'.pth'
		model_path_r = base_path+'right'+'epoch_'+str(epoch)+'.pth'
		save_model(left_model, model_path_l)
		save_model(right_model, model_path_r)
		# 	print("Epoch:", epoch, "Test Accuracy:", testacc.item()*100)
		# 	writer.add_scalar("Test Accuracy/3 epochs", testacc.item()*100, epoch)
		# if epoch%2==0:
		# 	testacc = test(model=model, dataloader=test_dataloader, output_classes=output_classes)
		# 	model_path = base_path+'epoch_'+str(epoch)+'.pth'
		# 	if testacc > best_acc:
		# 		best_acc = testacc
		# 		save_model(model, model_path)

		# 	print("Epoch:", epoch, "Test Accuracy:", testacc.item()*100)
		# 	writer.add_scalar("Test Accuracy/3 epochs", testacc.item()*100, epoch)

def test(model, dataloader, output_classes):
	model.eval()
	ypred = []
	ytrue = []
	
	for data in dataloader:
		pred  = model(data[0])
		ypred+=(torch.argmax(pred, dim=1)).tolist()
		ytrue+=(data[1]-1).tolist()
		
	acc = Accuracy(task="multiclass", num_classes=output_classes)
	accuracy = acc(tensor(ypred), tensor(ytrue))
	
	return accuracy

if __name__=="__main__":
	torch.backends.cudnn.enabled = False
	criterion = nn.CrossEntropyLoss()
	data_path = "56_dataset_85000_frames.csv"
	#test_path = "Data/test.csv"
	batch_size=1
	output_classes=5
	input_dim = 102
	embed_dim = 64
	epochs=100
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	train_dataset = ELE_Dataset(data_path, device, 'leftright', Window_Size=50)
	print("Train Dataset Size:", len(train_dataset))
	#test_dataset = ELE_Dataset(test_path, device, history)
	#print("Test Dataset Size:", len(test_dataset))
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	#test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
	left_model = ELE_Net(input_dim, embed_dim, output_classes).to(device)
	right_model = ELE_Net(input_dim, embed_dim, output_classes).to(device)

	optimizer_l = torch.optim.Adam(left_model.parameters(), lr=0.001)
	optimizer_r = torch.optim.Adam(right_model.parameters(), lr=0.001)
	
	base_path = "trained_models/"
	train(left_model, right_model, train_dataloader, epochs, 
		  optimizer_l, optimizer_r, criterion, output_classes, base_path, batch_size)
