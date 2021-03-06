import os
import numpy as np 
import torch
import torchvision.utils as vutils
from tqdm import tqdm
import matplotlib.pyplot as plt 

from dataloader import get_mnist_data, get_taxi_time_data, get_devanagari_data

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
		
		# pbar = tqdm(enumerate(test_loader), desc = 'Test Set Evaluation', total=iters_per_epoch)
		vis_counter = 0
		print('Test Set Evaluation')
		correct_predictions = 0.0
		for i, data in enumerate(test_loader):
			if args.data == 'mnist' or args.data == 'devanagari':
				x = data[0].view(-1, 784).numpy()
				y = data[1].numpy()
			elif args.data == 'taxi_time':
				x = data[0].numpy()
				y = data[1].numpy()
	        
			h1 = np.dot(x, W1) + b1 
			a1 = sigmoid(h1)
			h2 = np.dot(a1, W2) + b2
			a2 = sigmoid(h2)

			if args.data == 'mnist' or args.data == 'devanagari':
				correct_predictions += np.sum(np.argmax(a2, axis=1) == y)

				if args.visualize_eval:
					vis_counter += 1
					devanagari_character_list = np.array([key.split('_')[-1] for key in test_loader.dataset.class_to_idx.keys()])
					iters_per_epoch = np.ceil(len(test_loader.sampler) / test_loader.batch_size)
					print('Prediction: ', devanagari_character_list[ np.argmax(a2, axis=1).astype(np.int8) ])
					print('GT: ', devanagari_character_list[y])
					img_grid = vutils.make_grid(data[0])
					plt.close()
					plt.imshow(img_grid.permute(1,2,0).numpy())
					if vis_counter < 5:
						plt.show()
					else:
						exit(1)

			elif args.data == 'taxi_time':
				correct_predictions += np.sum((a2-y)**2)/100

		# print('Test Set Accuracy:(%)', correct_predictions*100/(test_loader.batch_size*iters_per_epoch))
		return correct_predictions*100/(len(test_loader.dataset))

	batch_size = args.batch_size
	test_batch_size = args.batch_size
	num_epochs = args.num_epochs
	LR = args.lr

	if args.optim == 'vanilla_gd':
		batch_size=None 

	if args.data == 'mnist':
		x_dim = 784
		y_dim = 10
		h_dim = 400
		W1 =  np.random.randn(x_dim, h_dim)*0.01
		b1 =  np.random.randn(1, h_dim)*0.01
		W2 =  np.random.randn(h_dim, y_dim)*0.01
		b2 =  np.random.randn(1, y_dim)*0.01

		train_loader, test_loader = get_mnist_data(batch_size, test_batch_size)

	elif args.data == 'taxi_time':
		x_dim = 20
		y_dim = 1
		h_dim = 300
		W1 =  np.random.randn(x_dim, h_dim)*0.01
		b1 =  np.random.randn(1, h_dim)*0.01
		W2 =  np.random.randn(h_dim, y_dim)*0.01
		b2 =  np.random.randn(1, y_dim)*0.01

		train_loader, test_loader = get_taxi_time_data(batch_size, test_batch_size)

	elif args.data == 'devanagari':
		x_dim = 784
		y_dim = 15
		h_dim = 400
		W1 =  np.random.randn(x_dim, h_dim)*0.01
		b1 =  np.random.randn(1, h_dim)*0.01
		W2 =  np.random.randn(h_dim, y_dim)*0.01
		b2 =  np.random.randn(1, y_dim)*0.01

		train_loader, test_loader = get_devanagari_data(batch_size, test_batch_size)

	else:
		raise NotImplementedError

	# Evaluation
	if args.eval_only:
		W1 = np.load(os.path.join(args.eval_param_dir, '{}_W1.npy'.format(args.optim)), allow_pickle=False)
		b1 = np.load(os.path.join(args.eval_param_dir, '{}_b1.npy'.format(args.optim)), allow_pickle=False)
		W2 = np.load(os.path.join(args.eval_param_dir, '{}_W2.npy'.format(args.optim)), allow_pickle=False)
		b2 = np.load(os.path.join(args.eval_param_dir, '{}_b2.npy'.format(args.optim)), allow_pickle=False)

		print('Evaluating on Test Set: ', predict())

		return

	training_loss_values = []
	test_accuracy_values = []

	for epoch in range(num_epochs):
		iters_per_epoch = np.ceil(len(train_loader.sampler) / train_loader.batch_size)
		pbar = tqdm(enumerate(train_loader), desc = 'Training Loss', total=iters_per_epoch)
	    
		epoch_loss = 0
		for i, data in pbar:

			if args.data == 'mnist' or args.data == 'devanagari':
				x = data[0].view(-1, 784).numpy()
				y = to_one_hot(data[1].numpy(), num_labels=y_dim)
			elif args.data == 'taxi_time':
				x = data[0].numpy()
				y = data[1].numpy()
	        
	        # Forward Pass
			h1 = np.dot(x, W1) + b1 
			a1 = sigmoid(h1)
			h2 = np.dot(a1, W2) + b2
			a2 = sigmoid(h2)
	        
			y_pred = a2
	        
	        # Calculate Loss
			if args.data == 'mnist' or args.data == 'devanagari':
				loss = -np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))/train_loader.batch_size
				da2 = -(y/y_pred - (1-y)/(1-y_pred))

			elif args.data == 'taxi_time':
				loss = np.mean((y_pred-y)**2)
				da2 = 2*(y_pred-y)

			epoch_loss += loss
	        
	        # Backward Pass
			dW2 = np.dot(a1.T, da2*a2*(1-a2))/train_loader.batch_size
			db2 = np.sum(da2, axis=0, keepdims=True)/train_loader.batch_size
	        
			dW1 = np.dot(x.T, np.dot(da2*a2*(1-a2), W2.T)*a1*(1-a1) )/train_loader.batch_size
			db1 = np.sum(np.dot(da2*a2*(1-a2), W2.T)*a1*(1-a1), axis=0, keepdims=True)/train_loader.batch_size
	        
			if args.optim == 'langevin_dynamics':
				W2 -= (0.5*(LR**2)*dW2 + LR*np.random.randn(*W2.shape))
				b2 -= (0.5*(LR**2)*db2 + LR*np.random.randn(*b2.shape))
				W1 -= (0.5*(LR**2)*dW1 + LR*np.random.randn(*W1.shape))
				b1 -= (0.5*(LR**2)*db1 + LR*np.random.randn(*b1.shape))
			else:
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

	# Save params
	os.makedirs(args.save_param_dir, exist_ok=True)
	np.save(os.path.join(args.save_param_dir, '{}_W1.npy'.format(args.optim)), W1, allow_pickle=False)
	np.save(os.path.join(args.save_param_dir, '{}_b1.npy'.format(args.optim)), b1, allow_pickle=False)
	np.save(os.path.join(args.save_param_dir, '{}_W2.npy'.format(args.optim)), W2, allow_pickle=False)
	np.save(os.path.join(args.save_param_dir, '{}_b2.npy'.format(args.optim)), b2, allow_pickle=False)

	os.makedirs(args.save_plots_dir, exist_ok=True)

	plt.plot(training_loss_values)
	plt.title('Training Loss for optim: {}'.format(args.optim))
	plt.ylabel('Training Loss')
	plt.xlabel('Epochs')
	plt.savefig(os.path.join(args.save_plots_dir, 'training_loss_{}.png'.format(args.optim)))

	plt.clf()
	plt.plot(test_accuracy_values)
	plt.title('Test Set MSE Loss for optim: {}'.format(args.optim))
	plt.ylabel('Test Set evaluation')
	plt.xlabel('Epochs')
	plt.savefig(os.path.join(args.save_plots_dir, 'test_accuracy_{}.png'.format(args.optim)))

	
