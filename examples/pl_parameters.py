#dataset
global dataset
global batch_size
global workers
global split
global height
global width
global combine_trainval
global num_instance
#model
global arch
global features
global dropout
#optimizer
global learningrate
global momentum
global weight_decay
global loss
#trainconfig
global resume
global evaluate_a
global epochs
global start_save
global seed
global print_freq
global distance_metric
global margin
global resumepath
#misc
global data_dir
global logs_dir

#oim
global oimscalar

#metric distance(triplet loss)
global distance_metric

#dataset
dataset = 'viper'
batch_size = 16
workers = 4
split = 0
height = 256  #inception=144
width = 128 #inception=56
combine_trainval = True
num_instance = ''

#model
arch = 'resnet50'
features = 128
dropout = 0.5

#optimizer
learningrate = 0.003
momentum = 0.9
weight_decay = 5e-4
loss = 'softmax'
margin = 0.5

#trainconfig
resume = False
#resumepath = '/media/saber/DATASET/reid-demo/open-reid/examples/logs/2018-09-04<reid.datasets.market1501.Market1501 object at 0x7f50cdd3e240>resnet50checkpoint.pth.tar'
evaluate_a = False
epochs = 3
start_save = 0
seed = 1
print_freq = 1
#data_dir='/media/saber/DATASET/reid-demo/open-reid/examples/data'
data_dir='/media/saber/DATASET/reid-demo/open-reid/examples/data'
logs_dir='/media/saber/DATASET/reid-demo/open-reid/examples/logs'



# metric learning
distance_metric = 'euclidean'

#oim
oim_scalar = 30
oim_momentum = 0.9