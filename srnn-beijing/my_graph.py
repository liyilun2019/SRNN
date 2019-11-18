import numpy as np
from helper import getVector, getMagnitudeAndDirection
import pandas as pd
import pickle

def read_pickle(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

class My_Graph():
    def __init__(self, seq_length,Wpath,Vpath):
        self.seq_length=seq_length;
        self.data_seq_raw = pd.read_csv(Vpath, header=None).values
        print(f'data_seq.shape={self.data_seq_raw.shape}');

        self.data_seq = self.z_score();

        self.frameNum=self.data_seq.shape[0]
        self.nodenum=self.data_seq.shape[1]
        self.nodes=[i for i in range(self.nodenum)]
        W = read_pickle(Wpath);
        print(f'W.shape={W.shape}')
        self.graph = []
        for o in self.nodes:
            for d in self.nodes:
                if(W[o][d]!=0):
                    self.graph.append((o,d));

    def z_score(self):
        self.mean = np.mean(self.data_seq_raw)
        self.std  = np.std(self.data_seq_raw)
        return (self.data_seq_raw - self.mean) / self.std

    def z_inverse(self,sequence):
        return sequence*self.std+self.mean

    def getSequence(self,start):
        retNodePresent = [self.nodes for c in range(self.seq_length)]
        retEdgePresent = [[] for c in range(self.seq_length)]
        numNodes = len(self.nodes)
        retNodes = np.zeros((self.seq_length, numNodes, 1))
        retEdges = np.zeros((self.seq_length, numNodes*numNodes, 2))  # Diagonal contains temporal edges
        for i in range(self.seq_length):
            retNodes[i,:,0]=self.data_seq[i+start]

        for u,v in self.graph:
            if(u==v):# 时
                for i in range(1,self.seq_length):
                    retEdges[i,u*numNodes+v,0]=self.data_seq[i+start-1,u]
                    retEdges[i,u*numNodes+v,1]=self.data_seq[i+start,v]
                    retEdgePresent[i].append((u,v));
            else:# 空
                for i in range(0,self.seq_length):
                    retEdges[i,u*numNodes+v,0]=self.data_seq[i+start,u]
                    retEdges[i,u*numNodes+v,1]=self.data_seq[i+start,v]  
                    retEdgePresent[i].append((u,v));
        return retNodes, retEdges, retNodePresent, retEdgePresent


    def getFrameNum(self):
        return self.frameNum

# filedir='../../STGCN_IJCAI-18/qtraffic/'
# grp = My_Graph(96,filedir+'W.pk',filedir+'traffic_data.csv')
# retNodes, retEdges, retNodePresent, retEdgePresent = grp.getSequence(10);
