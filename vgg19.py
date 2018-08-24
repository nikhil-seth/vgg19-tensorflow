""" VGG19 Model in Tensorflow without FC layers"""
"""
Function : 
model(image_height,image_weight,path)
model_info()
"""

import tensorflow as tf
import numpy as np
from scipy.io import loadmat

def model(image_height,image_width,path):
	"""
	Function Which creates Tensorflow Graph
	Arguments :
		path -> Path of Weight File (.mat)
	Returns :
		graph model in tensorflow
	"""

	def load_file(path='vgg19.mat'):
		"""
		Loads Weights File & returns Object of Numpy array
		"""
		file=loadmat(path)
		file=file['layers']
		print("Success load_file")
		return file

	def ret_layer_index(file):
		"""
		Takes file as input & returns a dictionary having name of layers with their code
		"""
		names={}
		for i in range(len(file[0])):
			print(file[0][i][0][0][0])
			names[file[0][i][0][0][0][0]]=i
		print("Success layer_index")
		return names
        
	def weight(layer_name):
		""" Asks for Layer Name & returns its weights & bias
		"""
		layer_no=names[layer_name]
		wb =file[0][layer_no][0][0][2]
		w=wb[0][0]
		b=wb[0][1]
		name=file[0][layer_no][0][0][0]
		assert name==layer_name
		print("Success weight")
		return w,b

	def conv_relu(prev_layer,layer_no,layer_name):
		W,b=weight(layer_name)
		W=tf.constant(W)
		b=tf.constant(np.reshape(b, (b.size)))
		l=tf.nn.conv2d(prev_layer,filter=W,strides=[1,1,1,1],padding='SAME') +b
		print("Success convrelu")
		return tf.nn.relu(l)

	def avg_pool(prev_layer):
		return tf.nn.avg_pool(prev_layer,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

	def load_graph():
		graph={}
		graph['input']   = tf.Variable(np.zeros((1, image_height, image_width,3)), dtype = 'float32')
		graph['conv1_1']  = conv_relu(graph['input'], 0, 'conv1_1')
		graph['conv1_2']  = conv_relu(graph['conv1_1'], 2, 'conv1_2')
		graph['avgpool1'] = avg_pool(graph['conv1_2'])
		graph['conv2_1']  = conv_relu(graph['avgpool1'], 5, 'conv2_1')
		graph['conv2_2']  = conv_relu(graph['conv2_1'], 7, 'conv2_2')
		graph['avgpool2'] = avg_pool(graph['conv2_2'])
		graph['conv3_1']  = conv_relu(graph['avgpool2'], 10, 'conv3_1')
		graph['conv3_2']  = conv_relu(graph['conv3_1'], 12, 'conv3_2')
		graph['conv3_3']  = conv_relu(graph['conv3_2'], 14, 'conv3_3')
		graph['conv3_4']  = conv_relu(graph['conv3_3'], 16, 'conv3_4')
		graph['avgpool3'] = avg_pool(graph['conv3_4'])
		graph['conv4_1']  = conv_relu(graph['avgpool3'], 19, 'conv4_1')
		graph['conv4_2']  = conv_relu(graph['conv4_1'], 21, 'conv4_2')
		graph['conv4_3']  = conv_relu(graph['conv4_2'], 23, 'conv4_3')
		graph['conv4_4']  = conv_relu(graph['conv4_3'], 25, 'conv4_4')
		graph['avgpool4'] = avg_pool(graph['conv4_4'])
		graph['conv5_1']  = conv_relu(graph['avgpool4'], 28, 'conv5_1')
		graph['conv5_2']  = conv_relu(graph['conv5_1'], 30, 'conv5_2')
		graph['conv5_3']  = conv_relu(graph['conv5_2'], 32, 'conv5_3')
		graph['conv5_4']  = conv_relu(graph['conv5_3'], 34, 'conv5_4')
		graph['avgpool5'] = avg_pool(graph['conv5_4'])
		return graph

	file=load_file(path)
	names=ret_layer_index(file)
	return load_graph()

def model_info():
	print("""Here is the detailed configuration of the VGG model:
        0 is conv1_1 (3, 3, 3, 64)
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu    
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax""")



