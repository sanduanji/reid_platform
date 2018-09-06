import pymongo
from pymongo import MongoClient
from examples.pl_parameters import *

conn = MongoClient('localhost',27017)
db = conn.reid
database = db.database
log = db.outlog
checkpoint = db.checkpoint
bestmodel = db.bestmodel


def save_log(time, timecost, arch, dataset, batch_size, loss, learningrate, epochs, distance_metric, mAP, rank1, rank5, rank10):
    r1={"time":time,"timecost":timecost, "dataset":dataset, "model":arch, "batchsize":batch_size, "loss":loss,\
        "lr":learningrate, "distance_metric":distance_metric, "epochs":epochs, "mAP":mAP, "Rank1":rank1,\
        "Rank5":rank5, "Rank10":rank10}
    rs = log.insert_one(r1)
    return rs


def sc_database(dataset):
        rs = database.find({"dataset":dataset})


def sc_log(cmc, rank1, rank5, rank10):
    for rs in log.find():
        print(rs)


def save_cpoint(arch, dataset, testtime, loss, batchsize, logs_dir):
    cpname = testtime+ str(dataset) + str(arch) + 'checkpoint.pth.tar'
    fpath = logs_dir
    r1 = {"name": cpname, "time": testtime, "arch":arch,"dataset":dataset,"loss":loss, "batchsize":batchsize, "resume_path": fpath}
    rs = checkpoint.insert_one(r1)
    return rs


def save_bmodel(arch, dataset, testtime, loss, batchsize, logs_dir):
    cpname = testtime+ str(dataset) + str(arch) + 'model_best.pth.tar'
    fpath = logs_dir + str(arch) + str(dataset) +testtime + 'model_best.pth.tar'
    r1 = {"name": cpname, "time": testtime, "arch":arch,"dataset":dataset,"loss":loss, "batchsize":batchsize, "resume_path": fpath}
    rs = checkpoint.insert_one(r1)
    return rs

