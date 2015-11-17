require 'cudnn'
require 'inn'
require 'hdf5'

function InceptionModule(name, inplane, outplane_a1x1, outplane_b3x3_reduce, outplane_b3x3, outplane_c5x5_reduce, outplane_c5x5, outplane_pool_proj)
  local a = nn.Sequential()
  local a1x1 = cudnn.SpatialConvolution(inplane, outplane_a1x1, 1, 1, 1, 1, 0, 0)
  a1x1.name = name .. '/1x1'
  a:add(a1x1)
  a:add(cudnn.ReLU(true))

  local b = nn.Sequential()
  local b3x3_reduce = cudnn.SpatialConvolution(inplane, outplane_b3x3_reduce, 1, 1, 1, 1, 0, 0)
  b3x3_reduce.name = name .. '/3x3_reduce'
  b:add(b3x3_reduce)
  b:add(cudnn.ReLU(true))
  local b3x3 = cudnn.SpatialConvolution(outplane_b3x3_reduce, outplane_b3x3, 3, 3, 1, 1, 1, 1)
  b3x3.name = name .. '/3x3'
  b:add(b3x3)
  b:add(cudnn.ReLU(true))

  local c = nn.Sequential()
  local c5x5_reduce = cudnn.SpatialConvolution(inplane, outplane_c5x5_reduce, 1, 1, 1, 1, 0, 0)
  c5x5_reduce.name = name .. '/5x5_reduce'
  c:add(c5x5_reduce)
  c:add(cudnn.ReLU(true))
  local c5x5 = cudnn.SpatialConvolution(outplane_c5x5_reduce, outplane_c5x5, 5, 5, 1, 1, 2, 2)
  c5x5.name = name .. '/5x5'
  c:add(c5x5)
  c:add(cudnn.ReLU(true))

  local d = nn.Sequential()
  d:add(cudnn.SpatialMaxPooling(3, 3, 1, 1, 1, 1))
  local d_pool_proj = cudnn.SpatialConvolution(inplane, outplane_pool_proj, 1, 1, 1, 1, 0, 0)
  d_pool_proj.name = name .. '/pool_proj'
  d:add(d_pool_proj)
  d:add(cudnn.ReLU(true))

  local module = nn.Sequential():add(nn.ConcatTable():add(a):add(b):add(c):add(d)):add(nn.JoinTable(2))
  return module
end

local model = nn.Sequential()

local conv1_7x7_s2 = cudnn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3)
conv1_7x7_s2.name = 'conv1/7x7_s2'
model:add(conv1_7x7_s2)
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())
model:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 1))

local conv_3x3_reduce = cudnn.SpatialConvolution(64, 64, 1, 1, 1, 1, 0, 0)
conv_3x3_reduce.name = 'conv2/3x3_reduce'
model:add(conv_3x3_reduce)
model:add(cudnn.ReLU(true))
local conv_3x3 = cudnn.SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
conv_3x3.name = 'conv2/3x3'
model:add(conv_3x3)
model:add(cudnn.ReLU(true))
model:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 1))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())

model:add(InceptionModule('inception_3a', 192, 64, 96, 128, 16, 32, 32))
model:add(InceptionModule('inception_3b', 256, 128, 128, 192, 32, 96, 64))

model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())

model:add(InceptionModule('inception_4a', 480, 192, 96, 208, 16, 48, 64))
model:add(InceptionModule('inception_4b', 512, 160, 112, 224, 24, 64, 64))
model:add(InceptionModule('inception_4c', 512, 128, 128, 256, 24, 64, 64))
model:add(InceptionModule('inception_4d', 512, 112, 144, 288, 32, 64, 64))
model:add(InceptionModule('inception_4e', 528, 256, 160, 320, 32, 128, 128))

model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2):ceil())

model:add(InceptionModule('inception_5a', 832, 256, 160, 320, 32, 128, 128))
model:add(InceptionModule('inception_5b', 832, 384, 192, 384, 48, 128, 128))

model:add(cudnn.SpatialAveragePooling(7, 7, 1, 1))
model:add(nn.View(-1, 1024))

model:add(nn.Dropout(0.4))

local classifier = nn.Linear(1024, 1000)
classifier.name = 'loss3/classifier'
model:add(classifier)

local paramsFile = hdf5.open('/host/googlenet.hdf5', 'r')
local moduleQueue = { model }
local touchedLayers = { }
for k1, v1 in ipairs(moduleQueue) do
  if v1.modules then
    for k2, v2 in ipairs(v1.modules) do
      table.insert(moduleQueue, v2)
    end
  end

  if v1.name then
    touchedLayers[v1.name] = true
    local layer = paramsFile:read(v1.name):all()
    if layer['000'] then
      v1.weight:copy(layer['000'])
    else
      print(v1.name .. ' has no weight')
    end
    if layer['001'] then
      v1.bias:copy(layer['001'])
    else
      print(v1.name .. ' has no bias')
    end
  end
end

paramsFile:close()

model = model:cuda()
torch.save('googlenet.t7', model)

local size = 224
local input = torch.CudaTensor(1, 3, size, size):fill(0)
local output = model:forward(input)
print(output:size())

local test = true
if test then
  local meanAccum = 0
  local count = 0
  local batchSize = 32
  local input = torch.CudaTensor(batchSize, 3, size, size)
  local targets = torch.LongTensor(batchSize)
  local dir = require 'pl.dir'
  local gm = require 'graphicsmagick'

  model:evaluate()

  local mean = torch.CudaTensor(input:size())
  mean[{{}, {1}, {}, {}}] = -104
  mean[{{}, {2}, {}, {}}] = -117
  mean[{{}, {3}, {}, {}}] = -123

  local dirs = dir.getdirectories('/data/imagenet_raw_images/256/val/')
  table.sort(dirs)
  local index = 1
  for k1, v1 in ipairs(dirs) do
    print(v1)
    for v2 in paths.iterfiles(v1) do
      local img = gm.Image():load(paths.concat(v1, v2), 256, 256):crop(size, size, (256 - size) / 2, (256 - size) / 2)
      local imgTensor = img:toTensor('float', 'BGR', 'DHW')
      input[{{index}, {}, {}, {}}]:copy(imgTensor)
      targets[{{index}}] = k1
      index = index + 1
      if index == batchSize + 1 then
        index = 1
        input:mul(255):add(mean)
        local output = model:forward(input)
        local _, order = output:sort(2)
        local accuracy = targets:eq(order[{{}, {order:size()[2]}}]:reshape(order:size()[1]):long()):float():mean()
        meanAccum = meanAccum + accuracy
        count = count + 1
        print(accuracy, meanAccum / count)
      end
    end
  end
end
