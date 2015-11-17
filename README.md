# googlenet-caffe2torch
Converts bvlc_googlenet.caffemodel to a Torch nn model.

Want to use the pre-trained GoogLeNet from the [BVLC Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) in Torch?  Do you not want to use Caffe as an additional dependency inside Torch?  Use these two scripts to build the network definition in Torch and copy the learned weights from the Caffe model.

* export_to_hdf5.py depends on a Caffe installation with pycaffe built.  This will parse the bvlc_googlenet.caffemodel protobuf and store each blob of weights as a dataset in an HDF5 file.
* googlenet.lua will build the network definition and then copy the weights from the HDF5 file built in the previous step.  Optionally, ```test = true``` can be set to verify that everything was done correctly and validation accuracy matches expectations (i.e. ~68% top-1 accuracy).

## FAQ

##### Why not train from scratch in Torch?
Because the work has already been done.  It's just not in the desired format.  I'd be thrilled to see the equivalent of the [BVLC Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) for Torch, but I didn't see anything.

##### Why not use [szagoruyko/torch-caffe-binding](https://github.com/szagoruyko/torch-caffe-binding)?
szagoruyko/torch-caffe-binding is fine if you need just the forward pass' output from your network or the gradient
of your input with respect to some loss, but if you wish to fine-tune your Caffe model with bells and whistles attached
in Torch, then you're out of luck.

##### Why not use [szagoruyko/loadcaffe](https://github.com/szagoruyko/loadcaffe)?
GoogLeNet is not a supported architecture.  I tried it and it produced a broken Torch model.  In particular, it serialized the four data paths from each Inception module rather than generating the nn.ConcatTable and nn.JoinTable.  It does not appear to respect "bottom" and "top" connection annotations.

##### Why HDF5 with this python script?  Why not parse the protobuf directly in lua / C?
Seemed easier.  There's [lua-protobuf](https://luarocks.org/modules/xavier-wang/lua-protobuf), but it seemed easier to use HDF5 in python and lua since both implementations get a lot of use and attention.  I would switch if Google published an official luarocks protobuf module.

##### Why didn't you implement the auxillary classifiers for GoogLeNet?
I'm lazy and didn't get around to it yet.  File a PR if you want to contribute.

##### I have a model other than GoogLeNet that I want to convert.  Any insight I can gain from these scripts?
* Make sure you set ceil_mode on your SpatialMaxPooling layers in Torch.  Caffe and Torch follow two different conventions regarding the output dimensions of their pooling implementations.  Caffe rounds up and Torch rounds down.  [[relevant commit]](https://github.com/soumith/cudnn.torch/commit/f58f3453c968d842efc1c4cadb019b4d8fe3e655)
* Torch's graphicsmagick module loads images from [0-1].  Caffe's GoogLeNet implementation was trained with values [0-255].  Scale them accordingly.
* Caffe uses OpenCV to load JPEGs and leaves them in their default BGR order.  You can use the Torch graphicsmagick module to load a BGR tensor directly.
* The mean specified in the GoogLeNet protobuf is already in BGR order.

##### You made a mistake.  How should I correct you?
File a github issue.

##### Can I get a gif of a dancing cat?
![yes](http://i.giphy.com/IcJ6n6VJNjRNS.gif)
