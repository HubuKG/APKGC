# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import os.path as osp
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm
import pdb
from mmkgc.data import TestDataLoader
from mmkgc.config import Tester


class Loss_log():
    def __init__(self, early_stop=10):
        self.acc = [0.]
        self.flag = 0
        self.early_stop_step = early_stop

        self.use_top_k_acc = 0

    def acc_init(self, topn=[1]):
        self.loss = []
        self.topn = topn
        self.use_top_k_acc = 1

    def update_acc(self, case):
        self.acc.append(case)

    def get_acc(self):
        return max(self.acc)

    def early_stop(self):
        if self.acc[-1] < max(self.acc):
            self.flag += 1
        else:
            self.flag = 0

        if self.flag >= self.early_stop_step:
            return True
        else:
            return False


class Trainer(object):

    def __init__(self,
                args=None,
                logger=None,
                model=None,
                data_loader=None,
                train_times=1000,
                alpha=0.5,
                use_gpu=True,
                opt_method="sgd",
                save_steps=None,
                checkpoint_dir=None,
                train_mode='adp',
                beta=0.5):

        self.work_threads = 8
        self.train_times = train_times
        self.args = args
        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir

        self.train_mode = train_mode
        self.beta = beta
        self.test_dataloader = TestDataLoader("./benchmarks/" + args.dataset + '/', "link")

        self.Loss_log = Loss_log(early_stop=args.early_stop)
        self.logger = logger



    def train_one_step(self, data):
        self.optimizer.zero_grad()
        if self.args.add_noise == 1 and self.args.noise_update == "epoch":
            self.model.model.update_noise()
        loss, _ = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer is not None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        self.logger.info("Finish initializing...")

        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            for data in self.data_loader:
                loss = self.train_one_step(data)
                res += loss
            training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
            

            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                self.logger.info(f"Epoch {epoch} | loss: {res}, saving...")
                tester = Tester.Tester(model=self.model.model, data_loader=self.test_dataloader, use_gpu=True)
                mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain=False)

                if mrr > self.Loss_log.get_acc():
                    self.best_model_wts = copy.deepcopy(self.model.model.state_dict())
                self.Loss_log.update_acc(mrr)
                if self.Loss_log.early_stop():
                    self.logger.info(" -------------------- Early Stop -------------------- ")
                    torch.save(self.best_model_wts, f"{self.args.save}-MRR{self.Loss_log.get_acc()}")
                    break
                
                self.model.model.train()

                self.logger.info(f"mrr:{mrr},\t mr:{mr},\t hit10:{hit10},\t hit3:{hit3},\t hit1:{hit1}")
                self.logger.info(f"{mrr}\t{mr}\t{hit10}\t{hit3}\t{hit1}")


    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
