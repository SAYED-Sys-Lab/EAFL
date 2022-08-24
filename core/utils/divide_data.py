# -*- coding: utf-8 -*-
from random import Random
import torch
from torch.utils.data import DataLoader
import numpy as np
import logging
import time
import random, csv
from argParser import args
#Ahmed - add new modules
from collections import OrderedDict
import os
import math

#set up the data generator to have consistent results
seed = 10
generator = torch.Generator()
generator.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = seed #torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partition data by trace or random"""

    def __init__(self, data, numOfClass=0, seed=10, isTest=False):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)

        self.data = data
        self.labels = self.data.targets
        self.args = args
        self.isTest = isTest
        np.random.seed(seed)

        self.data_len = len(self.data)
        self.task = args.task
        self.numOfLabels = numOfClass

        #Ahmed - set the number of samples per worker
        self.usedSamples = self.args.used_samples

        #Ahmed - introduce targets dict
        self.targets = OrderedDict()
        self.indexToLabel = {}

        # categarize the samples
        for index, label in enumerate(self.labels):
            if label not in self.targets:
                self.targets[label] = []
            self.targets[label].append(index)
            self.indexToLabel[index] = label
        self.classPerWorker = None

    def getNumOfLabels(self):
        return self.numOfLabels

    def getDataLen(self):
        return self.data_len

    def trace_partition(self, data_map_file):
        """Read data mapping from data_map_file. Format: <client_id, sample_name, sample_category, category_id>"""
        logging.info(f"Partitioning data by profile {data_map_file}...")

        clientId_maps = {}
        unique_clientIds = {}
        # load meta data from the data_map_file
        with open(data_map_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            read_first = True
            sample_id = 0

            for row in csv_reader:
                if read_first:
                    logging.info(f'Trace names are {", ".join(row)}')
                    read_first = False
                else:
                    client_id = row[0]

                    if client_id not in unique_clientIds:
                        unique_clientIds[client_id] = len(unique_clientIds)

                    clientId_maps[sample_id] = unique_clientIds[client_id]
                    sample_id += 1

        # Partition data given mapping
        self.partitions = [[] for _ in range(len(unique_clientIds))]

        for idx in range(len(self.data.data)):
            self.partitions[clientId_maps[idx]].append(idx)

    #Ahmed - add data mapping handlers (uniform, zipf, balanced) and class exclusion
    def partition_data_helper(self, num_clients, data_map_dir=None):
        tasktype = 'train' if not self.isTest else 'test'
        data_map_file = None
        if data_map_dir is not None:
            data_map_file = os.path.join(data_map_dir, tasktype + '.csv')
            #Ahmed - handle the case for reddit dataset where on IBEX mappings are stored on the metadata folder
            if args.data_set == 'reddit' or args.data_set == 'stackoverflow':
                data_map_dir = os.path.join(args.log_path, 'metadata', args.data_set, tasktype)
                data_map_file = os.path.join(data_map_dir,  'result_' + str(args.process_files_ratio) + '.csv')
        # Ahmed - introduce the mapping based on other methods rather than read mapping file to partition trace
        if self.isTest:
            if data_map_file is not None and num_clients >= args.total_worker:
                self.trace_partition(data_map_file)
            else:
                self.uniform_partition(num_clients=num_clients)
        else:
            if data_map_file is not None:
                self.trace_partition(data_map_file)
            elif self.args.partitioning <= 0 or self.isTest:
                self.uniform_partition(num_clients=num_clients)
            else:
                self.custom_partition(num_clients=num_clients)

    def uniform_partition(self, num_clients):
        # random uniform partition
        numOfLabels = self.getNumOfLabels()
        data_len = self.getDataLen()
        logging.info(
            f"Uniform partitioning data, {data_len} samples of {numOfLabels} labels on {num_clients} clients ...")

        indexes = list(range(data_len))
        self.rng.shuffle(indexes)

        for _ in range(num_clients):
            part_len = int(1. / num_clients * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def custom_partition(self, num_clients):
        # custom partition
        numOfLabels = self.getNumOfLabels()
        data_len = self.getDataLen()
        sizes = [1.0 / num_clients for _ in range(num_clients)]

        #get # of samples per worker
        # get number of samples per worker
        if self.usedSamples <= 0:
            self.usedSamples = int(data_len / num_clients)
        self.usedSamples = max(self.usedSamples, self.numOfLabels)

        #get targets
        targets = self.getTargets()
        keyDir = {key: int(key) for i, key in enumerate(targets.keys())}
        keyLength = [0] * numOfLabels
        for key in keyDir.keys():
            keyLength[keyDir[key]] = len(targets[key])

        logging.info(f"Custom partitioning {self.args.partitioning} data, {data_len} samples of {numOfLabels} labels on {num_clients} clients ...")

        ratioOfClassWorker = self.create_mapping(sizes)
        if ratioOfClassWorker is None:
            return self.uniform_partition(num_clients=num_clients)

        sumRatiosPerClass = np.sum(ratioOfClassWorker, axis=1)
        for worker in range(len(sizes)):
            ratioOfClassWorker[worker, :] = ratioOfClassWorker[worker, :] / float(sumRatiosPerClass[worker])

        # classPerWorker -> Rows are workers and cols are classes
        tempClassPerWorker = np.zeros([len(sizes), numOfLabels])
        # split the classes
        for worker in range(len(sizes)):
            self.partitions.append([])
            # enumerate the ratio of classes it should take
            for c in list(targets.keys()):
                # takeLength = min(int(ceil(keyLength[keyDir[c]] * ratioOfClassWorker[worker][keyDir[c]])), len(targets[c]))
                takeLength = min(math.ceil(self.usedSamples * ratioOfClassWorker[worker][keyDir[c]]), keyLength[keyDir[c]])
                self.rng.shuffle(targets[c])
                self.partitions[-1] += targets[c][0:takeLength]
                tempClassPerWorker[worker][keyDir[c]] += takeLength

            self.rng.shuffle(self.partitions[-1])

        #self.log_selection(tempClassPerWorker)
        del tempClassPerWorker

    def create_mapping(self, sizes):
        numOfLabels = self.getNumOfLabels()

        ratioOfClassWorker = None
        if self.args.partitioning == 1:
            ratioOfClassWorker = np.random.rand(len(sizes), numOfLabels).astype(np.float32)
        elif self.args.partitioning == 2:
            ratioOfClassWorker = np.random.zipf(self.args.zipf_param, [len(sizes), numOfLabels]).astype(np.float32)
        elif self.args.partitioning == 3:
            ratioOfClassWorker = np.ones((len(sizes), numOfLabels)).astype(np.float32)

        if self.args.filter_class > 0 or self.args.filter_class_ratio > 0:
            num_remove_class = self.args.filter_class if self.args.filter_class > 0 else round(numOfLabels * (1.0 - self.args.filter_class_ratio))
            for w in range(len(sizes)):
                # randomly filter classes by forcing zero samples
                wrandom = self.rng.sample(range(numOfLabels), num_remove_class)
                for wr in wrandom:
                    ratioOfClassWorker[w][wr] = 0.001

        logging.info("==== Class per worker p:{} s:{} l:{} c:{} ====\n {} \n".format(self.args.partitioning, len(sizes), numOfLabels,np.count_nonzero(ratioOfClassWorker),repr(ratioOfClassWorker)))
        return ratioOfClassWorker

    def getTargets(self):
        tempTarget = self.targets.copy()
        #TODO:why the temp targets are reshuffled each time getTargets is called?
        for key in tempTarget:
            self.rng.shuffle(tempTarget[key])
        return tempTarget

    def log_selection(self,classPerWorker):
        totalLabels = [0 for i in range(len(classPerWorker[0]))]
        logging.info("====Total # of workers is :{}, w/ {} labels, {}".format(len(classPerWorker), len(classPerWorker[0]), len(self.partitions)))
        for index, row in enumerate(classPerWorker):
            rowStr = ''
            numSamples = 0
            for i, label in enumerate(classPerWorker[index]):
                rowStr += '\t'+str(int(label))
                totalLabels[i] += label
                numSamples += label
            logging.info(str(index) + ':\t' + rowStr + '\t' + 'with sum:\t' + str(numSamples) + '\t' + repr(len(self.partitions[index])))
            logging.info("=====================================\n")
        logging.info("Total selected samples is: {}, with {}\n".format(str(sum(totalLabels)), repr(totalLabels)))
        logging.info("=====================================\n")


    def use(self, partition, istest):
        resultIndex = self.partitions[partition]

        exeuteLength = -1 if not istest else int(len(resultIndex) * self.args.test_ratio)
        resultIndex = resultIndex[:exeuteLength]
        self.rng.shuffle(resultIndex)

        return Partition(self.data, resultIndex)

    def getSize(self):
        # return the size of samples
        return {'size': [len(partition) for partition in self.partitions]}


def select_dataset(rank, partition, batch_size, isTest=False, collate_fn=None, seed=0):
    """Load data given client Id"""
    partition = partition.use(rank - 1, isTest)
    dropLast = False if isTest else True
    num_loaders = min(int(len(partition)/args.batch_size/2), args.num_loaders)
    if num_loaders == 0:
        time_out = 0
    else:
        time_out = 60

    if collate_fn is not None:
        return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn)
    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast)

    # if collate_fn is not None:
    #     return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn, worker_init_fn=seed_worker,  generator=generator)
    # return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, worker_init_fn=seed_worker, generator=generator)

    # if collate_fn is not None:
    #     return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, collate_fn=collate_fn, worker_init_fn=lambda id: np.random.seed(id + np.random.randint(10000)),  generator=generator)
    # return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast, worker_init_fn=lambda id: np.random.seed(id + np.random.randint(10000)), generator=generator)