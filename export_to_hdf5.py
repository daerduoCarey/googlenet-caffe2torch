from caffe.proto import caffe_pb2
import sys
import numpy as np
import h5py

net_param = caffe_pb2.NetParameter()
with open(sys.argv[1], 'r') as f:
  net_param.ParseFromString(f.read())

output_file = h5py.File(sys.argv[2], 'w')

for layer in net_param.layers:
  group = output_file.create_group(layer.name)
  for pos, blob in enumerate(layer.blobs):
    data = np.array(blob.data).reshape(blob.num, blob.channels, blob.height, blob.width)
    dataset = group.create_dataset('%03d' % pos, data=data)

output_file.close()
