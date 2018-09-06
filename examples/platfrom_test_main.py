from __future__ import print_function, absolute_import
import matplotlib.pyplot as plt

import argparse
import os.path as osp
import os

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
import datetime
import time
import os
from tkinter.filedialog import *
import shutil
import random

from reid import datasets
from reid import models
from reid.loss import TripletLoss
from reid.dist_metric import DistanceMetric
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.data.sampler import RandomIdentitySampler
from reid.evaluators import *

from reid.loss import OIMLoss

from examples.pl_createmodel import cr_oimmodel, cr_softmaxmodel, cr_tripletmodel
from examples.pl_parameters import *
from examples.pl_optimizer import softmax_op, triplet_op, oim_op
from examples.pl_getdata import getdata_sm, getdata_tl, getdata_oim
from examples.pl_mongodb import *

from pymongo import MongoClient

import matplotlib
#matplotlib.use('agg')
from PIL import Image

conn = MongoClient('localhost', 27017)
db = conn.reid
post = db.database
logout =db.outlog
ckpoint =db.checkpoint
bsmodel = db.bestmodel

global timecost, nowtime, savedt, resumepath, testtime
savedt = 'viper'
resumepath = ''
testtime = ''

y_loss = {}
y_loss['loss']=[]
#y_loss['prec']=[]
y_prec = {}
y_prec['rank1']=[]
y_prec['rank5']=[]
y_prec['rank10']=[]
y_prec['mAP']=[]
# Draw Curve
# ---------------------------
x_epoch = []
x_epoch1 = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="cmc")

# 进行新模型训练
def reid():
    global height
    global width
    global dataset
    global data_dir
    global nowtime,timecost
    global savelogdir
    global m_AP, Rank1, Rank5, Rank10

    mg_dataset = dataset
    mg_arch = arch
    mg_dismetric = distance_metric

    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['loss'], 'bo-', label='train')
#        ax0.plot(x_epoch, y_loss['prec'], 'ro-', label='val')
        if current_epoch == 0:
            ax0.legend()

    def draw_curve2(current_epoch):
        x_epoch1.append(current_epoch)
        ax1.plot(x_epoch1, y_prec['rank1'], 'ro-', label='rank1')
        ax1.plot(x_epoch1, y_prec['rank5'], 'go-', label='rank5')
        ax1.plot(x_epoch1, y_prec['rank10'], 'yo-', label='rank10')
        ax1.plot(x_epoch1, y_prec['mAP'], 'bo-', label='rank10')
        if current_epoch == 0:
            ax1.legend()

    start_time = time.time()
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    testtime = nowtime[:16]

    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True

    if dataset =='viper':
        savedt = 'viper'
    elif dataset == 'cuhk01':
        savedt = 'cuhk01'
    elif dataset == 'cuhk03':
        savedt = 'cuhk03'
    elif dataset == 'market1501':
        savedt = 'market1501'
    elif dataset == 'dukemtmc':
        savedt = 'dukemtmc'
    elif dataset == 'mars':
        savedt = 'mars'

    # Redirect print to both console and log file
    if not evaluate_a:
        sys.stdout = Logger(osp.join(logs_dir, 'log.txt'))

    # Create data loaders
    if height is None or width is None:
        height, width = (144, 56) if arch == 'inception' else \
            (256, 128)
    if loss == 'softmax':
        dataset, num_classes, train_loader, val_loader, test_loader = \
            getdata_sm(dataset, split, data_dir, height, width, batch_size, workers, combine_trainval)
    elif loss == 'triplet':
        dataset, num_classes, train_loader, val_loader, test_loader = \
            getdata_tl(dataset, split, data_dir, height, width, batch_size, workers, num_instance, combine_trainval)
    elif loss == 'oim':
        dataset, num_classes, train_loader, val_loader, test_loader = \
            getdata_oim(dataset, split, data_dir, height, width, batch_size, workers, combine_trainval)

    if loss == 'softmax':
        model = cr_softmaxmodel(arch, features, dropout, num_classes)
    elif loss == 'triplet':
        model = cr_tripletmodel(arch, dropout, features)
    elif loss == 'oim':
        model = cr_oimmodel(arch, features, dropout)

    # Load from checkpoint
    start_epoch = best_top1 = 0

    model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=distance_metric)

    # Evaluator
    evaluator = Evaluator(model)
    if evaluate_a:
        metric.train(model, train_loader)
        print("Validation:")
        evaluator.evaluate(val_loader, dataset.val, dataset.val, metric)
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)
        return

    # Criteron_choose
    if loss == 'softmax':
        criterion = nn.CrossEntropyLoss().cuda()
    elif loss == 'triplet':
        criterion = TripletLoss(margin=margin).cuda()
    elif loss == 'oim':
        criterion = OIMLoss(model.module.num_features, num_classes, scalar=oim_scalar,
                            momentum=oim_momentum).cuda()

    if loss == 'softmax':
        optimizer = softmax_op(model, learningrate, momentum, weight_decay)
    elif loss == 'triplet':
        optimizer = triplet_op(model, learningrate, weight_decay)
    elif loss == 'oim':
        optimizer = oim_op(model, learningrate, momentum, weight_decay)

    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 60 if arch == 'inception' else 40
        lr = learningrate * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    savelogdir = logs_dir + '/' + mg_arch + mg_dataset + "checkpoint.pth.tar"
    print(savelogdir)

    # Start training
    for epoch in range(start_epoch, epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)
        if epoch < start_save:
            continue
        top1, top5, top10, mAP = evaluator.evaluate(val_loader, dataset.val, dataset.val)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(logs_dir, (testtime + savedt + str(arch)+'checkpoint.pth.tar')))
        fpath = osp.join(logs_dir, (testtime + savedt + str(arch) + 'checkpoint.pth.tar'))
        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))
        y_loss['loss'].append(trainer.losses.avg)
#        y_loss['prec'].append(trainer.precisions.avg)

        y_prec['rank1'].append(top1)
        y_prec['rank5'].append(top5)
        y_prec['rank10'].append(top10)
        y_prec['mAP'].append(mAP)
        draw_curve(epoch)
        draw_curve2(epoch)
        fig.savefig(os.path.join(logs_dir, (testtime + savedt + str(arch) + 'train.jpg')))

    save_cpoint(str(arch), savedt, testtime, loss, batch_size, fpath)

    # Final test
    print('Test with best model:')
    bsmodelpath = testtime + savedt+ str(arch) + 'model_best.pth.tar'
    checkpoint = load_checkpoint(osp.join(logs_dir, bsmodelpath))
    model.module.load_state_dict(checkpoint['state_dict'])
    metric.train(model, train_loader)
    Rank1, Rank5, Rank10, m_AP= evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)

    timecost = round(time.time() - start_time)
    timecost = str(datetime.timedelta(seconds=timecost))
    save_log(nowtime, timecost, mg_arch, mg_dataset, batch_size, loss, learningrate, epochs, mg_dismetric, format(m_AP,'.2%'),\
             format(Rank1,'.2%'), format(Rank5,'.2%'), format(Rank10,'.2%'))
    plt.show()
    endout()


# 载入预训练模型
def runcheck(*args):
    global dataset
    print(resumepath)

    def loadckpoint(pth):
        for loss in ckpoint.find({"resume_path": pth},
                                 {"name": 0, "time": 0, "arch": 0, "dataset": 0, "batchsize": 0, "resume_path": 0,
                                  "_id": 0}):
            closs = str(loss)[10:-2]
        for arch in ckpoint.find({"resume_path": pth},
                                 {"name": 0, "time": 0, "loss": 0, "dataset": 0, "batchsize": 0, "resume_path": 0,
                                  "_id": 0}):
            carch = (str(arch)[10:-2])
        for dataset in ckpoint.find({"resume_path": pth},
                                    {"name": 0, "time": 0, "loss": 0, "arch": 0, "batchsize": 0, "resume_path": 0,
                                     "_id": 0}):
            cdataset = str(dataset)[13:-2]
        for batchsize in ckpoint.find({"resume_path": pth},
                                      {"name": 0, "time": 0, "loss": 0, "arch": 0, "dataset": 0, "resume_path": 0,
                                       "_id": 0}):
            cbatchsize = str(batchsize)[14:-1]

        return carch, cdataset, cbatchsize, closs

    ckarch, ckdataset, cbatchsize, ckloss = loadckpoint(resumepath)
    cbatchsize =int(cbatchsize)

    if ckloss == 'softmax':
        dataset, num_classes, train_loader, val_loader, test_loader = \
            getdata_sm(ckdataset, split, data_dir, height, width, cbatchsize, workers, combine_trainval)
    elif ckloss == 'triplet':
        dataset, num_classes, train_loader, val_loader, test_loader = \
            getdata_tl(ckdataset, split, data_dir, height, width, cbatchsize, workers, combine_trainval)
    elif ckloss == 'oim':
        dataset, num_classes, train_loader, val_loader, test_loader = \
            getdata_oim(ckdataset, split, data_dir, height, width, cbatchsize, workers, combine_trainval)

    if ckloss == 'softmax':
        model = cr_softmaxmodel(ckarch, features, dropout, num_classes)
    elif ckloss == 'triplet':
        model = cr_tripletmodel(ckarch, features, dropout, num_classes)
    elif ckloss == 'oim':
        model = cr_oimmodel(ckarch, features, dropout)

    # Load from checkpoint
    checkpoint = load_checkpoint(resumepath)
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']
    best_top1 = checkpoint['best_top1']
    print("=> Start epoch {}  best top1 {:.1%}"
            .format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=distance_metric)

    # Evaluator
    evaluator = Evaluator(model)

    metric.train(model, train_loader)
    print("Validation:")
    evaluator.evaluate(val_loader, dataset.val, dataset.val, metric)
    print("Test:")
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)
    return

#训练新模型
#---------------------------
def trainning():

    top = tk.Tk()
    top.title("Python GUI")
    top.geometry("550x300+300+100")

    sysfont = tkFont.Font(size=18)
    menubar = tk.Menu(top)

    fmenu1 = tk.Menu(top)
    for item in ['New', 'Open', 'Save', 'Save as']:
        fmenu1.add_command(label=item)

    fmenu2 = tk.Menu(top)
    for item in ['Copy', 'Paste', 'Cut']:
        fmenu2.add_command(label=item)

    fmenu3 = tk.Menu(top)
    for item in ['Datebase', 'Output']:
        fmenu3.add_command(label=item)

    fmenu4 = tk.Menu(top)
    for item in ["Version", "About"]:
        fmenu4.add_command(label=item)

    menubar.add_cascade(label="file", menu=fmenu1)
    menubar.add_cascade(label="edit", menu=fmenu2)
    menubar.add_cascade(label="database", menu=fmenu3)
    menubar.add_cascade(label="about", menu=fmenu4)

    top['menu'] = menubar

    def modelchoose(*args):
        global arch
        arch = str(ModelChosen.get())

    def datachoose(*args):
        global dataset
        dataset = str(DatabaseChosen.get())
        get_dtn = dataset

    def losschoose(*args): #*args
        global loss
        loss = str(LossChosen.get())

    def para_confirm():
        global batch_size
        global learningrate
        global epochs
        global margin
        global momentum
        global oimscalar

        if (trainbatch.get()):
            batch_size = int(trainbatch.get())
        if (lr.get()):
            learningrate = float(lr.get())
        if (ep.get()):
            epochs = int(ep.get())
        if (marginchose.get()):
            margin = float(marginchose.get())
        if (momentumchose.get()):
            momentum = float(momentumchose.get())
        if (scalarchose.get()):
            oimscalar = int(scalarchose.get())
        print("train_batchsize: %s\nlearning rate: %s\nepoch: %s\n" % (batch_size, learningrate, epochs))

    ttk.Label(top, text="Platform").grid(column=0, row=0, pady=2)
    ttk.Label(top, text="Loss").grid(column=0, row=1, pady=2)
    ttk.Label(top, text="Database").grid(column=0, row=2, pady=2)
    ttk.Label(top, text="Model").grid(column=0, row=3, pady=2)
    ttk.Label(top, text="GPU_device_number").grid(column=0, row=4, pady=2)
    ttk.Label(top, text="Batchsize").grid(column=0, row=5, pady=2)
    ttk.Label(top, text="Learning rate").grid(column=0, row=6, pady=2)
    ttk.Label(top, text="Epoch").grid(column=0, row=7, pady=2)
    ttk.Label(top, text="Margin(Only use in Triplet Loss)").grid(column=0, row=8, pady=2)
    ttk.Label(top, text="Momentum(Only use in OIM Loss)").grid(column=0, row=9, pady=2)
    ttk.Label(top, text="Scalar(Only use in OIM Loss)").grid(column=0, row=10, pady=2)

    ttk.Label(top, text="Default: Pytorch").grid(column=2, row=0, pady=2)
    ttk.Label(top, text="Default: Softmax").grid(column=2, row=1, pady=2)
    ttk.Label(top, text="Default: Viper").grid(column=2, row=2, pady=2)
    ttk.Label(top, text="Default: resnet50").grid(column=2, row=3, pady=2)
    ttk.Label(top, text="Default: 0").grid(column=2, row=4, pady=2)
    ttk.Label(top, text="Default: 64").grid(column=2, row=5, pady=2)
    ttk.Label(top, text="Default: 0.003").grid(column=2, row=6, pady=2)
    ttk.Label(top, text="Default: 50").grid(column=2, row=7, pady=2)
    ttk.Label(top, text="Default: 0").grid(column=2, row=8, pady=2)
    ttk.Label(top, text="Default: 0.9").grid(column=2, row=9, pady=2)
    ttk.Label(top, text="Default: 30").grid(column=2, row=10, pady=2)

    trainbatch = ttk.Entry(top)
    lr = ttk.Entry(top)
    ep = ttk.Entry(top)
    marginchose = ttk.Entry(top)  #Only use in triplet loss
    momentumchose = ttk.Entry(top)  #Only use in oim loss
    scalarchose = ttk.Entry(top)  #Only use in oim loss

    trainbatch.grid(row=5, column=1, pady=1)
    lr.grid(row=6, column=1, pady=1)
    ep.grid(row=7, column=1, pady=1)
    marginchose.grid(row=8, column=1, pady=1)   #Only use in triplet loss
    momentumchose.grid(row=9,column=1,pady=1)  #Only use in oim loss
    scalarchose.grid(row=10,column=1,pady=1)  #Only use in oim loss

    ttk.Button(top, text='Training', command=reid).grid(row=11, column=1, pady=4)
    ttk.Button(top, text='Parameters Confirm', command=para_confirm).grid(row=11, column=0, pady=4)

    Platform = tk.StringVar()
    PlatformChosen = ttk.Combobox(top, width=18, textvariable=Platform)
    PlatformChosen['values'] = ('Tensorflow', 'Pytorch', 'Mxnet')
    PlatformChosen.grid(column=1, row=0, pady=5)
    PlatformChosen.current(1)

    Loss = tk.StringVar()
    LossChosen = ttk.Combobox(top, width=18, textvariable=Loss)
    LossChosen['values'] = ('softmax', 'triplet', 'oim')
    LossChosen.grid(column=1, row=1, pady=5)
    LossChosen.current(0)
    LossChosen.bind("<<ComboboxSelected>>", losschoose)

    Database = tk.StringVar()
    DatabaseChosen = ttk.Combobox(top, width=18, textvariable=Database)
    DatabaseChosen['values'] = ('viper', 'market1501', 'cuhk01', 'cuhk03', 'dukemtmc', 'msmt17', 'grid')  # 设置下拉列表的值
    DatabaseChosen.grid(column=1, row=2, pady=5)
    DatabaseChosen.current(0)
    DatabaseChosen.bind("<<ComboboxSelected>>", datachoose)

    Model = tk.StringVar()
    ModelChosen = ttk.Combobox(top, width=18, textvariable=Model)
    ModelChosen['values'] = (
        'resnet50', 'resnet18', 'resnet34', 'resnet101', 'resnet152', 'resnet50M', 'densenet121', 'squeezenet',  'shufflenet',
         'xception', 'nasnsetmobile', 'mudeep', 'hacnn','inception')
    ModelChosen.grid(column=1, row=3, pady=5)
    ModelChosen.current(0)
    ModelChosen.bind("<<ComboboxSelected>>", modelchoose)

    GPU_device = tk.StringVar()
    GPU_deviceChosen = ttk.Combobox(top, width=18, textvariable=GPU_device)
    GPU_deviceChosen['values'] = ('0', '1', '2', '3')
    GPU_deviceChosen.grid(column=1, row=4, pady=5)
    GPU_deviceChosen.current(0)

    top.mainloop()

#加载已训练过的模型
#---------------------------------
def loadcheckpoint():
    load = tk.Tk()
    load.title("Python GUI")
    load.geometry("600x400+300+100")

    sysfont = tkFont.Font(size=18)
    menubar = tk.Menu(load)

    fmenu1 = tk.Menu(load)
    for item in ['New', 'Open', 'Save', 'Save as']:
        fmenu1.add_command(label=item)

    fmenu2 = tk.Menu(load)
    for item in ['Copy', 'Paste', 'Cut']:
        fmenu2.add_command(label=item)

    fmenu3 = tk.Menu(load)
    for item in ['Datebase', 'Output']:
        fmenu3.add_command(label=item)

    fmenu4 = tk.Menu(load)
    for item in ["Version", "About"]:
        fmenu4.add_command(label=item)

    menubar.add_cascade(label="file", menu=fmenu1)
    menubar.add_cascade(label="edit", menu=fmenu2)
    menubar.add_cascade(label="database", menu=fmenu3)
    menubar.add_cascade(label="about", menu=fmenu4)

    load['menu'] = menubar

    def checkpoint(*args):
        global cp_dir
        global resumepath

        oct(os.stat('/media/saber/').st_mode)[-3:]
        cp_dir = askopenfilename(title="Choose Checkpoint Dataset", initialdir='/media/saber/DATASET/reid-demo/open-reid/examples/logs')
        if cp_dir:
            resumepath = cp_dir
#        print(resumepath)

    def bestmodel(*args):
        global bp_dir
        global resumepath


        oct(os.stat('/media/saber/').st_mode)[-3:]
        bp_dir = askopenfilename(title="Choose Bestmodel Dataset", initialdir='/media/saber/DATASET/reid-demo/open-reid/examples/logs')
        if bp_dir:
            resumepath = bp_dir

    ttk.Button(load, text='Loading CheckPoint', command=checkpoint).grid(row=0, column=0, pady=4)
    ttk.Button(load, text='Loading Best Models', command=bestmodel).grid(row=0, column=1, pady=4)
    ttk.Button(load, text='Continue Training', command=runcheck).grid(row=0, column=2, pady=4)

    load.mainloop()


def endout():
    global timecost, nowtime, get_dtn

    eout = tk.Tk()
    eout.title("Training Output")
    eout.geometry("600x400+300+100")

    listbox_out = Listbox(eout, width=195, height=200)
    listbox_out.grid(row=2, column=0, columnspan=4, rowspan=4,
                          padx=5, pady=5, sticky=W + E + S + N)

    listbox_out.insert(END, "Start Time:")
    listbox_out.insert(END, nowtime)
    listbox_out.insert(END, "Training Time:")
    listbox_out.insert(END, timecost)
    listbox_out.insert(END, "Dataset:")
    listbox_out.insert(END, savedt)
    listbox_out.insert(END, "Model:")
    listbox_out.insert(END, arch)
    listbox_out.insert(END, "Batchsize:")
    listbox_out.insert(END, batch_size)
    listbox_out.insert(END, "Loss Function:")
    listbox_out.insert(END, loss)
    listbox_out.insert(END, "Learning Rate:")
    listbox_out.insert(END, learningrate)
    listbox_out.insert(END, "Training Epochs:")
    listbox_out.insert(END, epochs)
    listbox_out.insert(END, "Distance Metric:")
    listbox_out.insert(END, distance_metric)
    listbox_out.insert(END, "mAP:")
    listbox_out.insert(END, format(m_AP,'.2%'))
    listbox_out.insert(END, "Rank1:")
    listbox_out.insert(END, format(Rank1,'.2%'))
    listbox_out.insert(END, "Rank5:")
    listbox_out.insert(END, format(Rank5,'.2%'))
    listbox_out.insert(END, "Rank10:")
    listbox_out.insert(END, format(Rank10,'.2%'))
    eout.mainloop()


def maintabel():
    main = tk.Tk()
    main.title("Reid Platform")
    main.geometry("450x300")

    sysfont = tkFont.Font(size=18)
    menubar = tk.Menu(main)

    fmenu1 = tk.Menu(main)
    for item in ['New', 'Open', 'Save', 'Save as']:
        fmenu1.add_command(label=item)

    fmenu2 = tk.Menu(main)
    for item in ['Copy', 'Paste', 'Cut']:
        fmenu2.add_command(label=item)

    fmenu3 = tk.Menu(main)
    for item in ['Datebase', 'Output']:
        fmenu3.add_command(label=item)

    fmenu4 = tk.Menu(main)
    for item in ["Version", "About"]:
        fmenu4.add_command(label=item)

    menubar.add_cascade(label="file", menu=fmenu1)
    menubar.add_cascade(label="edit", menu=fmenu2)
    menubar.add_cascade(label="database", menu=fmenu3)
    menubar.add_cascade(label="about", menu=fmenu4)

    main['menu'] = menubar

#上传数据集
#------------------------------
    def upload_dataset():
        oct(os.stat('/media/saber/').st_mode)[-3:]    #赋予打开文件权限
        filename = askopenfilename(title="Choose Uploading Dataset", initialdir='/home/saber')
        old = filename
        new = "/media/saber/DATASET/dataset"
        shutil.copy(old, new)
        filepath = new + '/' + filename.split('/')[-1]
        print(filename)
        print(filepath)
        k = os.path.exists(filepath)
        if k:
            print("Uploading Success")
        else:
            print("Uploading Failed")

#查看数据集
#-------------------------------
    def viewdataset():
        vdata = tk.Tk()
        vdata.title("Viewing Output")
        vdata.geometry("600x300")
        global photochose
        photochose = ''

        def viewphoto():
            global photochose
            photochose = PhotoChosen.get()


            if photochose == 'cuhk03':
                os.system('nautilus /media/saber/DATASET/dataset/reid_image/cuhk03')
            elif photochose == 'cuhk01':
                os.system('nautilus /media/saber/DATASET/dataset/reid_image/cuhk01')
            elif photochose == 'dukemtmc':
                os.system('nautilus /media/saber/DATASET/dataset/reid_image/DukeMTMC-reID_evaluation-master')
            elif photochose == 'market1501':
                os.system('nautilus /media/saber/DATASET/dataset/reid_image/market1501')
            elif photochose == 'mtmc17':
                os.system('nautilus /media/saber/DATASET/dataset/reid_image/MSMT17_V1')
            elif photochose == 'viper':
                os.system('nautilus /media/saber/DATASET/dataset/reid_image/viper')
            elif photochose == 'mars':
                os.system('nautilus /media/saber/DATASET/dataset/reid_video/MARS-v160809')
            elif photochose == 'prid_2011':
                os.system('nautilus /media/saber/DATASET/dataset/reid_video/prid_2011')
            elif photochose == 'i-LIDS-VID':
                os.system('nautilus /media/saber/DATASET/dataset/reid_video/i-LIDS-VID')
            elif photochose == '' :
                tk.messagebox.showinfo(title='Warning', message='You have not chose a dataset')


        Photo = tk.StringVar()
        PhotoChosen = ttk.Combobox(vdata, width=18, textvariable=Photo)
        PhotoChosen['values'] = ('viper', 'cuhk03', 'duckmtmc','mtmc17', 'cuhk01','market1501')
        PhotoChosen.place(x=220, y=90, width=120, height=25)
#       PhotoChosen.current(0)
#       PhotoChosen.bind("<<ComboboxSelected>>", photoon)

        tk.Label(vdata, text = 'Choose a dataset: ').place(x=60, y=90, width=150 ,height=25)

        tk.Button(vdata, text='View Dataset', command=viewphoto).place(x=360, y=90, width=120, height=25)

        vdata.mainloop()

#查看训练结果
    def viewoutput():
        vout =tk.Tk()
        vout.title("Viewing Output")
        vout.geometry("1450x600+30+30")
        vout.rowconfigure(1, weight=1)
        vout.rowconfigure(2, weight=8)

        def searchout():
            for out in logout.find():
                ans = str(out)
                ans1 = ans[45:-1]
                ans2 = ans1.strip(':')
                ans3 = ans2.replace('\'', '')
                ans4 = ans3.replace(',', '')
                listbox_filename.insert(END, ans4)

        button = Button(vout, text="search out", command=searchout)
        button.grid(sticky=W + N, row=1, column=1, padx=5, pady=5)

        listbox_filename = Listbox(vout, width=210)
        listbox_filename.grid(row=2, column=0, columnspan=4, rowspan=4,
                              padx=5, pady=5, sticky=W + E + S + N)
        vout.mainloop()

    l =ttk.Label(main, text='Choose the function you need').place(x=90,y=30,width=250, height=25)
    frame = Frame(height=700, width=1300,bg="Aqua").pack(expand=YES,fill=BOTH)

    tk.Button(main, text='Training New Model', command=trainning).place(x=90, y=60, width=250, height=25)
    tk.Button(main, text='Loading CheckPoint Model', command=loadcheckpoint).place(x=90, y=90, width=250, height=25)
    tk.Button(main, text='Viewing Training Outputs', command=viewoutput).place(x=90, y=120, width=250, height=25)
    tk.Button(main, text='Viewing Reid Dataset', command=viewdataset).place(x=90, y=150, width=250, height=25)
    tk.Button(main, text='Uploading Reid Dataset', command=upload_dataset).place(x=90, y=180, width=250, height=25)

    main.mainloop()


maintabel()








