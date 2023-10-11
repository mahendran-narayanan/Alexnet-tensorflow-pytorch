import argparse
import tensorflow as tf
import torch

# def model_tf():
# 	create = []
# 	create.append(tf.keras.layers.Conv2D(96,11,strides=4,input_shape=(256,256,3),activation='relu'))
# 	create.append(tf.keras.layers.Conv2D(256,5,activation='relu'))
# 	return tf.keras.Sequential(*create)

class TfNN(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.conv1 = tf.keras.layers.Conv2D(96,11,strides=4,activation='relu')
		self.conv2 = tf.keras.layers.Conv2D(256,5,activation='relu')

	def call(self,x):
		x = self.conv1(x)
		x = self.conv2(x)
		return x

class TorchNN(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(3, 96, kernel_size=11, stride=4)
		self.conv2 = torch.nn.Conv2d(96, 256, kernel_size=5)
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		return x

def model_torch():
	return TorchNN()

def model_tf():
	return TfNN()

def main(args):
	# print(args.model)
	if args.model=='tf':
		print('Model will be created in Tensorflow')
		model = model_tf()
		model.build(input_shape=(None,256,256,3))
		model.summary()
	else:
		print('Model will be created in Pytorch')
		model = model_torch()
		print(model)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create alexnet model in Tensorflow or Pytorch package')
	parser.add_argument('--model',
	                    default='tf',
	                    # const='all',
	                    # nargs='?',
	                    choices=['tf', 'torch'],
	                    help='Model created on Tensorflow, Pytorch (default: %(default)s)')
	args = parser.parse_args()
	main(args)