import os
import numpy as np 
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt 

from dataloader import get_mnist_data

def to_one_hot(y, num_labels):
	one_hot = np.zeros((len(y), num_labels))
	for i in range(len(y)):
		one_hot[i, y[i]] = 1.
	return one_hot

def sigmoid(x):
	return 1/(1+np.exp(-x))

def train(args):

	# Make this a separate function later
	def predict():
		iters_per_epoch = np.ceil(len(test_loader.sampler) / test_loader.batch_size)
		pbar = tqdm(enumerate(test_loader), desc = 'Test Set Evaluation', total=iters_per_epoch)
		correct_predictions = 0.0
		for i, data in pbar:
			x = data[0].view(-1, 784).numpy()
			y = data[1].numpy()
	        
			h1 = np.dot(x, W1) + b1 
			a1 = sigmoid(h1)
			h2 = np.dot(a1, W2) + b2
			a2 = sigmoid(h2)

			correct_predictions += np.sum(np.argmax(a2, axis=1) == y)

		# print('Test Set Accuracy:(%)', correct_predictions*100/(test_loader.batch_size*iters_per_epoch))
		return correct_predictions*100/(test_loader.batch_size*iters_per_epoch)


	batch_size = args.batch_size
	num_epochs = args.num_epochs
	lr = args.lr

	if args.data == 'mnist':
		x_dim = 784
		y_dim = 10
		h_dim = 400
		W1 =  np.random.randn(x_dim, h_dim)*0.01
		b1 =  np.random.randn(1, h_dim)*0.01
		W2 =  np.random.randn(h_dim, y_dim)*0.01
		b2 =  np.random.randn(1, y_dim)*0.01

		train_loader, test_loader = get_mnist_data(batch_size)
	else:
		raise NotImplementedError

	training_loss_values = []
	test_accuracy_values = []
	for epoch in range(num_epochs):
		iters_per_epoch = np.ceil(len(train_loader.sampler) / train_loader.batch_size)
		pbar = tqdm(enumerate(train_loader), desc = 'Training Loss', total=iters_per_epoch)
	    
		epoch_loss = 0
		for i, data in pbar:
			x = data[0].view(-1, 784).numpy()
			y = to_one_hot(data[1].numpy(), num_labels=y_dim)
	        
	        # Forward Pass
			h1 = np.dot(x, W1) + b1 
			a1 = sigmoid(h1)
			h2 = np.dot(a1, W2) + b2
			a2 = sigmoid(h2)
	        
			y_pred = a2
	        
	        # Calculate Loss
			loss = -np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))/batch_size
			epoch_loss += loss
	        
	        # Backward Pass
			da2 = -(y/y_pred - (1-y)/(1-y_pred))
	        
			dW2 = np.dot(a1.T, da2*a2*(1-a2))/batch_size
			db2 = np.sum(da2, axis=0, keepdims=True)/batch_size
	        
			dW1 = np.dot(x.T, np.dot(da2*a2*(1-a2), W2.T)*a1*(1-a1) )/batch_size
			db1 = np.sum(np.dot(da2*a2*(1-a2), W2.T)*a1*(1-a1), axis=0, keepdims=True)/batch_size
	        
			W2 -= lr*dW2
			b2 -= lr*db2
			W1 -= lr*dW1
			b1 -= lr*db1

			pbar.set_description('Epoch: {}'.format(epoch)) # Printing Batch Loss here slows down training 
	        
		print('Average Epoch Loss:', epoch_loss/iters_per_epoch)
		training_loss_values.append(epoch_loss/iters_per_epoch)
		test_accuracy_values.append(predict())

	# print('Final Evaluation on Test Set:')
	# predict()

	plt.plot(training_loss_values)
	plt.title('Training Loss')
	plt.ylabel('Average BCE Loss')
	plt.xlabel('Epochs')
	plt.savefig('training_loss.png')

	plt.clf()
	plt.plot(test_accuracy_values)
	plt.title('Test Set Accuracy')
	plt.ylabel('Correct classification percentage')
	plt.xlabel('Epochs')
	plt.savefig('test_accuracy.png')

	
