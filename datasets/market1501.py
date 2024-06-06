# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

import pandas as pd

from .bases import BaseImageDataset
from collections import defaultdict
import pickle


attribute_names = {
    'gender': {
        1: 'man',
        2: 'woman'
    },
    'hair': {
        1: 'short hair',
        2: 'long hair'
    },
    'sleeve': {
        1: 'long sleeve',
        2: 'short sleeve',
    },
    'lower body cloth length': {
        1: 'long',
        2: 'short'
    },
    'lower body cloth type': {
        1: 'dress',
        2: 'pants'
    },
    'hat': {
        1: 'without hat',
        2: 'wearing hat'
    },
    'backpack': {
        1: 'without backpack',
        2: 'carrying backpack'
    },
    'bag': {
        1: 'without bag',
        2: 'carrying bag'
    },
    'handbag': {
        1: 'without handbag',
        2: 'carrying handbag'
    },
    'age': {
        1: 'young',
        2: 'teenager',
        3: 'adult',
        4: 'old'
    },
    'upper body cloth color': {
        0: 'black',
        1: 'blue',
        2: 'gray',
        3: 'green',
        4: 'purple',
        5: 'red',
        6: 'white',
        7: 'yellow'
    },
    'lower body cloth color': {
        0: 'black',
        1: 'blue',
        2: 'brown',
        3: 'gray',
        4: 'green',
        5: 'pink',
        6: 'purple',
        7: 'white',
        8: 'yellow'
    }
}


class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'Market-1501-v15.09.15'

    def __init__(self, root='', attributes_dir='', verbose=True, pid_begin = 0, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.attributes_dir = attributes_dir
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.attributes_train_path = osp.join(self.attributes_dir, 'attributes_train.csv')

        self._check_before_run()
        self.pid_begin = pid_begin
        attributes_train = pd.read_csv(self.attributes_train_path, index_col=0)
        train, attributes_train = self._process_dir(self.train_dir, relabel=True, annos=attributes_train)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.attributes_train = attributes_train
        self.attribute_names = attribute_names
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.attributes_train_path):
            raise RuntimeError("'{}' is not available".format(self.attributes_train_path))

    def _process_dir(self, dir_path, relabel=False, annos=None):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if annos is not None and relabel:
            annos = annos.rename(index=pid2label)
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 0))
            
        if annos is not None:
            return dataset, annos
        else:
            return dataset
