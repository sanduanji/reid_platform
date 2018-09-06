from __future__ import print_function, absolute_import
import os.path as osp

from reid.utils.data import Dataset
from reid.utils.osutils import mkdir_if_missing
from reid.utils.serialization import write_json

import shutil

dataset_dir = 'msmt17'


class MSMT17(Dataset):
    url = 'https://docs.google.com/forms/d/e/1FAIpQLScIGhLvB2GzIXjX1oFW0tNUWxkbK2l0fYG5Q9vX93ls2BVsQw/viewform?usp=sf_link'
#    md5 = '2f93496f9b516d1ee5ef51c1d5e7d601'

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(MSMT17, self).__init__(root, split_id=split_id)
        self.dataset_dir = '/media/saber/DATASET/reid-demo/open-reid/examples/data/msmt17/msmt17'
        self.list_train_path = osp.join(self.dataset_dir, 'MSMT17_V1/list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'MSMT17_V1/list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'MSMT17_V1/list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'MSMT17_V1/list_gallery.txt')
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'MSMT17_V1/train')
        self.test_dir = osp.join(self.dataset_dir, 'MSMT17_V1/test')

        self.download()

        identities = []
        all_pids = {}

        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        def process_dir(dir_path, list_path):
            with open(list_path, 'r') as txt:
                lines = txt.readlines()
            dataset = []
            pid_container = set()
            pids = set()
            for img_idx, img_info in enumerate(lines):
                img_path, pid = img_info.split(' ')
                pid = int(pid)  # no need to relabel
                camid = int(img_path.split('_')[2])
                img_path = osp.join(dir_path, img_path)
                dataset.append((img_path, pid, camid))
                pids.add(pid)
                pid_container.add(pid)
                if pid >= len(identities):
#                    assert pid == len(identities)
                    identities.append([[] for _ in range(15)])  # 15 camera views 从1号摄像头到15号,而列表从0开始,所以这里可用16
#                    print(identities)
#                print(identities[pid][camid-1])
                camid =camid-1
#                print(camid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, camid, len(identities[pid][camid])))
#                print(fname)
                identities[pid][camid].append(fname)
                shutil.copy(img_path, osp.join(images_dir, fname))
            num_imgs = len(dataset)
            num_pids = len(pid_container)
            # check if pid starts from 0 and increments with 1
            for idx, pid in enumerate(pid_container):
                assert idx == pid, "See code comment for explanation"
            return dataset, num_pids, num_imgs, pids



        train, num_train_pids, num_train_imgs, trainval_pids = process_dir(self.train_dir, self.list_train_path)
        # val, num_val_pids, num_val_imgs = self._process_dir(self.train_dir, self.list_val_path)
        query, num_query_pids, num_query_imgs, query_pids = process_dir(self.test_dir, self.list_query_path)
        gallery, num_gallery_pids, num_gallery_imgs, gallery_pids = process_dir(self.test_dir, self.list_gallery_path)

        self.train = train
        self.query = query
        self.gallery = gallery

#        trainval_pids = num_train_pids
#        gallery_pids = num_query_pids
#        query_pids = num_gallery_pids
        num_total_pids = num_train_pids + num_query_pids

        assert query_pids <= gallery_pids
#        assert trainval_pids.isdisjoint(gallery_pids)

        # Save meta information into a json file
        meta = {'name': 'MSMT17', 'shot': 'multiple', 'num_cameras': 15,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import re
        import hashlib
        import shutil
        import tarfile
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'MSMT17_V1.tar.gz')
        if osp.isfile(fpath):
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} " "to {}".format(self.url, fpath))

        # Extract the file
        exdir = osp.join(raw_dir, 'MSMT17_V1')
        if not osp.isdir(exdir):
            print("Extracting tar.zp file")
            with tarfile.open(fpath) as t:
                t.extractall(path=raw_dir)

        # Format






