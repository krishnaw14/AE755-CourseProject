import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch

def get_mnist_data(batch_size):
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
	mnist_train_data = MNIST('data/', train=True, download=True, transform=transform)
	mnist_test_data = MNIST('data/', train=False, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(mnist_train_data, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=batch_size, shuffle=True)

	return train_loader, test_loader