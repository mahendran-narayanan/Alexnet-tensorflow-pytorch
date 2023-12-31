import argparse
import tensorflow as tf
import torch


class TfNN(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.conv1 = tf.keras.layers.Conv2D(96,11,strides=4,activation='relu')
		self.conv2 = tf.keras.layers.Conv2D(256,5,activation='relu')
		self.conv3 = tf.keras.layers.Conv2D(384,3,activation='relu')
		self.conv4 = tf.keras.layers.Conv2D(384,3,activation='relu')
		self.conv5 = tf.keras.layers.Conv2D(256,3,activation='relu')
		self.flat = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(4096,activation='relu')
		self.dense2 = tf.keras.layers.Dense(4096,activation='relu')
		self.classifier = tf.keras.layers.Dense(1000,activation='sigmoid')

	def call(self,x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		x = self.flat(x)
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.classifier(x)
		return x

class Alexnet_torch(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(3, 96, kernel_size=11, stride=4)
		self.relu1 = torch.nn.ReLU()
		self.conv2 = torch.nn.Conv2d(96, 256, kernel_size=5)
		self.relu2 = torch.nn.ReLU()
		self.conv3 = torch.nn.Conv2d(256, 384, kernel_size=3)
		self.relu3 = torch.nn.ReLU()
		self.conv4 = torch.nn.Conv2d(384, 384, kernel_size=3)
		self.relu4 = torch.nn.ReLU()
		self.conv5 = torch.nn.Conv2d(256, 256, kernel_size=3)
		self.relu5 = torch.nn.ReLU()
		self.flatten = torch.nn.Flatten()
		self.dense1 = torch.nn.Linear(4096,4096)
		self.dense2 = torch.nn.Linear(4096,4096)
		self.classifier = torch.nn.Linear(4096,1000)
		
	def forward(self, x):
		x = self.relu1(self.conv1(x))
		x = self.relu2(self.conv2(x))
		x = self.relu3(self.conv3(x))
		x = self.relu4(self.conv4(x))
		x = self.relu5(self.conv5(x))
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.classifier(x)
		return x

def model_torch():
	return Alexnet_torch()

def model_tf():
	return TfNN()

def main(args):
	if args.model=='tf':
		print('Model will be created in Tensorflow')
		model = model_tf()
		model.build(input_shape=(None,224,224,3))
		model.summary()
	else:
		print('Model will be created in Pytorch')
		model = model_torch()
		print(model)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create alexnet model in Tensorflow or Pytorch package')
	parser.add_argument('--model',
	                    default='tf',
	                    choices=['tf', 'torch'],
	                    help='Model created on Tensorflow, Pytorch (default: %(default)s)')
	args = parser.parse_args()
	main(args)