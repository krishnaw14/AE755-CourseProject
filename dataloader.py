import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import torch

import pandas as pd 
import numpy as np 

def get_mnist_data(batch_size):
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
	mnist_train_data = MNIST('data/', train=True, download=True, transform=transform)
	mnist_test_data = MNIST('data/', train=False, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(mnist_train_data, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=batch_size, shuffle=True)

	return train_loader, test_loader

def get_taxi_time_data(batch_size, data_path='data/taxi_time.csv'):

	dataset = TaxiTimeData(data_path)
	dataset = LimitDataset(dataset, 10000)
	dataset_len = len(dataset)
	train_data, test_data = torch.utils.data.random_split(dataset, [int(dataset_len*0.85), dataset_len-int(dataset_len*0.85)])

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

	return train_loader, test_loader

class TaxiTimeData(Dataset):

	def __init__(self, data_path):

		df = pd.read_csv(data_path)

		#remove unneeded columns
		remove_colums = ['flight', 'acType', 'actual_landing_time', 'actual_inblock_time', 
		'scheduled_time_off', 'estimated_inblock_time', 'calculated_inblock_time', 'date', 'week_of_year', 'day_of_week', 
		'minutes_of_day', 'minute_of_week']

		weather_features = ['average_wind_speed', 'precipitation', 'average_temperature', 'direction_of_fastest_2_minute_wind', 
		'direction_of_fastest_5_second_wind', 'fastest_2_minute_wind_speed', 'fog,_ice_fog,_or_freezing_fog_(incl_heavy_fog)', 
		'heavy_fog_or_heaving_freezing_fog', 'thunder', 'smoke_or_haze']

		traffic_features = ['traffic_metric_runway', 'traffic_metric_runway_1','traffic_metric_runway_2', 'traffic_metric_runway_3',
		'traffic_metric_runway_5','traffic_metric_runway_6', 'traffic_metric_runway_7',
		'traffic_metric_runway_8', 'traffic_metric_runway_9','traffic_metric_runway_10']

		airport_features = ['runway', 'stand']

		output_features = ['t_minutes']

       # Nomalize the data
		df[weather_features] = (df[weather_features] - df[weather_features].mean())/df[weather_features].std()
		df[traffic_features] = (df[traffic_features] - df[traffic_features].mean())/df[traffic_features].std()
		df[output_features] = (df[output_features] - df[output_features].min())/(df[output_features].max()-df[output_features].min())

       # Remove Unnecessary Entries
		df = df[np.isnan(df[weather_features].values)==False]
		df = df[np.isnan(df[traffic_features].values)==False]
		df = df[np.isnan(df[output_features].values)==False]

		self.weather_features = df[weather_features].values
		self.traffic_features = df[traffic_features].values

		self.input = np.concatenate((self.weather_features, self.traffic_features), axis=1)
		self.output = df[output_features].values

	def __len__(self):
		return self.input.shape[0]

	def __getitem__(self, idx):
		return self.input[idx], self.output[idx]

class LimitDataset(Dataset):
	def __init__(self, dataset, n):
		self.n = n
		self.dataset = dataset
	def __len__(self):
		return self.n
	def __getitem__(self, i):
		return self.dataset[i]





