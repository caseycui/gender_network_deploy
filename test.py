import os
import numpy as np
import matplotlib.pyplot as plt


caffe_root = '/home/case/machine_learning/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
#from caffe.proto import caffe_pb2

import time
#from google.protobuf import text_format


#Loading the mean image
mean_filename='/home/case/machine_learning/HW/models/gender/pretrained_model/mean.binaryproto'
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]

#Loading the gender network
#net = caffe.Net('/home/case/machine_learning/HW/models/gender/pretrained_model/deploy_gender.prototxt', '/home/case/machine_learning/HW/models/gender/pretrained_model/gender_net.caffemodel', caffe.TEST)
gender_net_pretrained = '/home/case/machine_learning/HW/models/gender/pretrained_model/gender_net.caffemodel'
gender_net_prototxt = '/home/case/machine_learning/HW/models/gender/pretrained_model/deploy_gender.prototxt'
gender_net = caffe.Classifier(gender_net_prototxt,gender_net_pretrained,mean=mean,channel_swap=(2,1,0),raw_scale=255,image_dims=(256,256))
#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_mean('data',np.load().mean(1).mean(1))
#transformer.set_transpose('data',(2,0,1)) #(channel, width, height)
#transformer.set_channel_swap('data',(2,1,0)) #(RGB)->BGR
#transformer.set_raw_scale('data',255.0)

#Labels
gender_list=['Male','Female']

#Reading and plotting the input image
example_image = '/home/case/machine_learning/HW/models/gender/deploy_example/example_image.jpg'
input_image = caffe.io.load_image(example_image)
_ = plt.imshow(input_image)

#Gender prediction
prediction = gender_net.predict([input_image])

print('predicted gender: {}'.format(gender_list[prediction[0].argmax()]))


#reshape input layer (we could change the batch size)
#net.blobs['data'].reshape(1,3,227,227)
#load the image in the data layer
#im = caffe.io.load_image('.jpg')
#net.blobs['data'].data[...] = transformer.preprocess('data',im)
#forwarding
#out = net.forward()
#predict result
#print out['prob'].argmax()
#Get feature from "layer", e.g., layer='fc7'
#fea = np.copy(net.blobs[layer].data)



#if __name__ == "__main__":
  
    #caffemodel = '/home/case/machine_learning/HW/models/gender/pretrained_model/gender_net.caffemodel'
    #deploy_file = '/home/case/machine_learning/HW/models/gender/pretrained_model.deploy_gender.prototxt'
    #mean_file = None

    #gpu = False
    #net = Deep_net(caffemodel, deploy_file, mean_file,gpu)
    #net.test()