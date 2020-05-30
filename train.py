import os
import numpy as np 
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt 

from dataloader import get_mnist_data, get_taxi_time_data

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
			if args.data == 'mnist':
				x = data[0].view(-1, 784).numpy()
				y = to_one_hot(data[1].numpy(), num_labels=y_dim)
			elif args.data == 'taxi_time':
				x = data[0].numpy()
				y = data[1].numpy()
	        
			h1 = np.dot(x, W1) + b1 
			a1 = sigmoid(h1)
			h2 = np.dot(a1, W2) + b2
			a2 = sigmoid(h2)

			if args.data == 'mnist':
				correct_predictions += np.sum(np.argmax(a2, axis=1) == y)
			elif args.data == 'taxi_time':
				correct_predictions += np.sum((a2-y)**2)/100

		# print('Test Set Accuracy:(%)', correct_predictions*100/(test_loader.BATCH_SIZE*iters_per_epoch))
		return correct_predictions*100/(test_loader.batch_size*iters_per_epoch)


	BATCH_SIZE = args.BATCH_SIZE
	NUM_EPOCHS = args.NUM_EPOCHS
	LR = args.LR

	if args.data == 'mnist':
		x_dim = 784
		y_dim = 10
		h_dim = 400
		W1 =  np.random.randn(x_dim, h_dim)*0.01
		b1 =  np.random.randn(1, h_dim)*0.01
		W2 =  np.random.randn(h_dim, y_dim)*0.01
		b2 =  np.random.randn(1, y_dim)*0.01

		train_loader, test_loader = get_mnist_data(BATCH_SIZE)

	elif args.data == 'taxi_time':
		x_dim = 20
		y_dim = 1
		h_dim = 300
		W1 =  np.random.randn(x_dim, h_dim)*0.01
		b1 =  np.random.randn(1, h_dim)*0.01
		W2 =  np.random.randn(h_dim, y_dim)*0.01
		b2 =  np.random.randn(1, y_dim)*0.01

		train_loader, test_loader = get_taxi_time_data(BATCH_SIZE)

	else:
		raise NotImplementedError

	training_loss_values = []
	test_accuracy_values = []

	for epoch in range(NUM_EPOCHS):
		iters_per_epoch = np.ceil(len(train_loader.sampler) / train_loader.batch_size)
		pbar = tqdm(enumerate(train_loader), desc = 'Training Loss', total=iters_per_epoch)
	    
		epoch_loss = 0
		for i, data in pbar:
			if args.data == 'mnist':
				x = data[0].view(-1, 784).numpy()
				y = to_one_hot(data[1].numpy(), num_labels=y_dim)
			elif args.data == 'taxi_time':
				x = data[0].numpy()
				y = data[1].numpy()
				# import pdb; pdb.set_trace()
	        
	        # Forward Pass
			h1 = np.dot(x, W1) + b1 
			a1 = sigmoid(h1)
			h2 = np.dot(a1, W2) + b2
			a2 = sigmoid(h2)
	        
			y_pred = a2
	        
	        # Calculate Loss
			if args.data == 'mnist':
				loss = -np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))/BATCH_SIZE
				da2 = -(y/y_pred - (1-y)/(1-y_pred))

			elif args.data == 'taxi_time':
				loss = np.mean((y_pred-y)**2)
				da2 = 2*(y_pred-y)

			# import pdb; pdb.set_trace()
			epoch_loss += loss
	        
	        # Backward Pass
			dW2 = np.dot(a1.T, da2*a2*(1-a2))/BATCH_SIZE
			db2 = np.sum(da2, axis=0, keepdims=True)/BATCH_SIZE
	        
			dW1 = np.dot(x.T, np.dot(da2*a2*(1-a2), W2.T)*a1*(1-a1) )/BATCH_SIZE
			db1 = np.sum(np.dot(da2*a2*(1-a2), W2.T)*a1*(1-a1), axis=0, keepdims=True)/BATCH_SIZE
	        
			W2 -= LR*dW2
			b2 -= LR*db2
			W1 -= LR*dW1
			b1 -= LR*db1

			pbar.set_description('Epoch: {}'.format(epoch)) # Printing Batch Loss here slows down training 
	        
		print('Average Epoch Loss:', epoch_loss/iters_per_epoch)
		training_loss_values.append(epoch_loss/iters_per_epoch)
		test_accuracy_values.append(predict())

	print('Final Evaluation on Test Set:')
	print(predict())

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

	
