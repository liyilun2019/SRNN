
import os
import pickle
import numpy as np
import random
import torch

def getL2Loss(outputs,targets,nodesPresent,pred_length):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution
    params:
    outputs: Torch variable containing tensor of shape seq_length x numNodes x output_size
    targets: Torch variable containing tensor of shape seq_length x numNodes x input_size
    nodesPresent : A list of lists, of size seq_length. Each list contains the nodeIDs that are present in the frame
    '''
    loss = 0
    counter = 0
    seq_length = outputs.size()[0]
    obs_length = seq_length - pred_length
    framenum=-1
    nodeIDs = nodesPresent[framenum]

    for nodeID in nodeIDs:

        loss = loss + torch.pow((outputs[framenum,nodeID,0]-targets[framenum,nodeID,0]),2)
        counter = counter + 1

    return loss

#    if counter != 0:
#        return loss / counter
#    else:
#        return loss


def get_MAPE(outputs,targets,nodesPresent):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution
    params:
    outputs: Torch variable containing tensor of shape seq_length x numNodes x output_size
    targets: Torch variable containing tensor of shape seq_length x numNodes x input_size
    '''
    
    counter = 0
    mape =[]

    print(outputs.shape)
    print(targets.shape)

    for framenum in range(len(outputs)):
        nodeIDs = nodesPresent
        tmp=0
        counter = 0
        for nodeID in nodeIDs:
            o=outputs[framenum,nodeID,0]
            t=targets[framenum,nodeID,0]
            tmp+=np.abs(o-t)/t
            counter = counter + 1
        if(counter!=0):
            tmp/=counter
        mape.append(tmp)

    return mape


def fake_edge(nodes, tstep, edgesPresent):
    numNodes = nodes.size()[1]
    edges = (torch.zeros(numNodes * numNodes, 2)).cuda()
    for edgeID in edgesPresent:
        nodeID_a = edgeID[0]
        nodeID_b = edgeID[1]
        new_edge=(torch.zeros(1, 2)).cuda()

        if nodeID_a == nodeID_b:
            # Temporal edge
            new_edge[0,0] = nodes[tstep - 1, nodeID_a, 0]
            new_edge[0,1] = nodes[tstep, nodeID_b, 0]

            edges[nodeID_a * numNodes + nodeID_b, :] = new_edge[0,:]
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)
        else:
            # Spatial edge
            new_edge[0,0] = nodes[tstep, nodeID_a, 0]
            new_edge[0,1] = nodes[tstep, nodeID_b, 0]

            edges[nodeID_a * numNodes + nodeID_b, :] = new_edge[0,:]
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)

    return edges
