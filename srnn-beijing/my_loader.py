import os
import pickle
import numpy as np
import random
import math

class My_DataPointer():
    def __init__(self,frameNum,seq_len,batch_size):
        self.totle=frameNum-seq_len;
        trainend=round(self.totle*0.6)
        self.train=[i for i in range(0,trainend,seq_len-1)]
        validend=trainend+math.floor(self.totle*0.15)
        self.val =[i for i in range(trainend,validend,seq_len-1)]
        self.test=[i for i in range(validend,self.totle,seq_len-1)]
        
        self.trainpointer=0
        self.valpointer=0
        self.testpointer=0
        self.batch_size=batch_size

        print(f"train set = {len(self.train)}")
        print(f"val set = {len(self.val)}")
        print(f"test set = {len(self.test)}")

    def train_reset(self):
        random.shuffle(self.train)
        self.trainpointer=0

    def val_reset(self):
        self.valpointer=0

    def test_reset(self):
        self.testpointer=0

    def get_batch(self):
        self.trainpointer+=self.batch_size
        return self.train[self.trainpointer-self.batch_size:self.trainpointer]

    def get_batch_val(self):
        self.valpointer+=self.batch_size
        return self.train[self.valpointer-self.batch_size:self.valpointer]

    def num_batches(self):
        return math.floor(len(self.train)/self.batch_size)

    def valid_num_batches(self):
        return math.floor(len(self.val)/self.batch_size)

    def test_num(self):
        return len(self.test)

    def get_test(self):
        return self.test