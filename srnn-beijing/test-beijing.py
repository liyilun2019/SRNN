'''
Test script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 2nd April 2017
'''


import os
import pickle
import argparse
import time

import torch
from torch.autograd import Variable
torch.cuda.set_device(1)


import numpy as np
from model import SRNN
from my_graph import My_Graph
from my_loader import My_DataPointer
from my_utils import getL2Loss,get_MAPE,fake_edge


def main():

    parser = argparse.ArgumentParser()
    # Observed length of the trajectory parameter
    parser.add_argument('--obs_length', type=int, default=16,
                        help='Observed length of the trajectory')
    # Predicted length of the trajectory parameter
    parser.add_argument('--pred_length', type=int, default=8,
                        help='Predicted length of the trajectory')
    # Model to be loaded
    parser.add_argument('--epoch', type=int, default=1,
                        help='Epoch of model to be loaded')

    # Parse the parameters
    sample_args = parser.parse_args()

    # Save directory
    save_directory = '../save-beijing-16/'

    # Define the path for the config file for saved args
    with open(os.path.join(save_directory, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
        args=saved_args

    # Initialize net
    net = SRNN(saved_args)
    net.cuda()

    checkpoint_path = os.path.join(save_directory, 'srnn_model_'+str(sample_args.epoch)+'.tar')

    if os.path.isfile(checkpoint_path):
        print('Loading checkpoint')
        checkpoint = torch.load(checkpoint_path)
        # model_iteration = checkpoint['iteration']
        model_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        print('Loaded checkpoint at {}'.format(model_epoch))

    # Dataset to get data from
    # dataset = [sample_args.test_dataset]

    # dataloader = DataLoader(1, sample_args.pred_length + sample_args.obs_length, dataset, True, infer=True)

    # dataloader.reset_batch_pointer()

    filedir='../../beijing/'

    grp = My_Graph(sample_args.obs_length+sample_args.pred_length,filedir+'W.pk',filedir+'traffic_data.csv')

    my_datapt=My_DataPointer(grp.getFrameNum(),sample_args.obs_length+sample_args.pred_length,1)
    

    # Construct the ST-graph object
    # stgraph = ST_GRAPH(1, sample_args.pred_length + sample_args.obs_length)

    atten_res=[]
    mape_mean=[]
    loss=0
    # Variable to maintain total error
    total_error = 0
    final_error = 0
    cnt=0;

    # num = my_datapt.test_num();
    num=1

    fo=open(save_directory+'output.txt',"w")

    with torch.no_grad():
        for e in range(num):
            for st in my_datapt.get_test():
                start = time.time()

                # Get the next st
                # x, _, frameIDs, d = dataloader.next_batch(randomUpdate=False)

                # Construct ST graph
                # stgraph.readGraph(x)

                nodes, edges, nodesPresent, edgesPresent = grp.getSequence(st)
                nodes = Variable(torch.from_numpy(nodes).float() ).cuda()
                edges = Variable(torch.from_numpy(edges).float() ).cuda()

                # obs_nodes, obs_edges, obs_nodesPresent, obs_edgesPresent = nodes[:sample_args.obs_length], edges[:sample_args.obs_length], nodesPresent[:sample_args.obs_length], edgesPresent[:sample_args.obs_length]
                # ret_nodes, ret_attn = multiPred(obs_nodes, obs_edges, obs_nodesPresent, obs_edgesPresent, sample_args, net)

                numNodes = nodes.size()[1]
                hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()
                cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()


                # ret_nodes, _, _, _, _, _ = net(nodes, edges, nodesPresent, edgesPresent,
                #                                  hidden_states_node_RNNs, hidden_states_edge_RNNs,
                #                                  cell_states_node_RNNs, cell_states_edge_RNNs)

                ret_nodes,_=multiPred(nodes,edges,nodesPresent,edgesPresent,sample_args,net)

                ret_nodes_numpy=ret_nodes.cpu().numpy()
                ret_nodes_numpy=grp.z_inverse(ret_nodes_numpy)
                nodes_numpy=nodes.cpu().numpy()
                nodes_numpy=grp.z_inverse(nodes_numpy)

                # atten_res.append(ret_attn)
                mape = get_MAPE(ret_nodes_numpy[sample_args.obs_length:], nodes_numpy[sample_args.obs_length:], nodesPresent[sample_args.obs_length-1])

                end = time.time()

                print(f"    mape:{mape}")
                fo.write(f"    mape:{mape}\n")
                with open("../log-beijing-16/log.py","w") as f:
                    f.write(f"frame=[];\n")
                    for i in range(ret_nodes_numpy.shape[0]):
                        f.write(f"frame.append([\n")
                        for node in range(ret_nodes_numpy.shape[1]):
                            f.write(f"({ret_nodes_numpy[i,node,0]},{nodes_numpy[i,node,0]}),")
                        f.write("]);\n")
                if(len(mape_mean)==0):
                    mape_mean=mape
                else:
                    for i in range(len(mape_mean)):
                        mape_mean[i]+=mape[i]


    for i in range(len(mape_mean)):
        mape_mean[i]=mape_mean[i]/my_datapt.test_num();
        print(f"time step{i}:mape={mape_mean[i]}\n")
        fo.write(f"time step{i}:mape={mape_mean[i]}\n")
    with open('../log-beijing-16/attention_out.txt',"w") as f:
        f.write(str(atten_res));



def multiPred(nodes, edges, nodesPresent, edgesPresent, args, net):
    # Number of nodes
    numNodes = nodes.size()[1]

    with torch.no_grad():
        # Initialize hidden states for the nodes
        
        # Propagate the observed length of the trajectory
        # for tstep in range(args.obs_length-1):
            # Forward prop
            # out_obs, h_nodes, h_edges, c_nodes, c_edges, _ = net(nodes[tstep].view(1, numNodes, 1), edges[tstep].view(1, numNodes*numNodes, 2), [nodesPresent[tstep]], [edgesPresent[tstep]], h_nodes, h_edges, c_nodes, c_edges)
            # loss_obs = Gaussian2DLikelihood(out_obs, nodes[tstep+1].view(1, numNodes, 2), [nodesPresent[tstep+1]])

        # Initialize the return data structures
        ret_nodes = Variable(torch.zeros(args.obs_length + args.pred_length+1, numNodes, 1)).cuda()
        ret_nodes[:args.obs_length, :, :] = nodes[:args.obs_length].clone()

        ret_edges = Variable(torch.zeros((args.obs_length + args.pred_length+1), numNodes * numNodes, 2)).cuda()
        ret_edges[:args.obs_length, :, :] = edges[:args.obs_length].clone()

        ret_attn = []

        # Propagate the predicted length of trajectory (sampling from previous prediction)
        for tstep in range(args.obs_length-1, args.pred_length + args.obs_length-1):
            h_nodes = Variable(torch.zeros(numNodes, net.args.human_node_rnn_size)).cuda()
            h_edges = Variable(torch.zeros(numNodes * numNodes, net.args.human_human_edge_rnn_size)).cuda()
            c_nodes = Variable(torch.zeros(numNodes, net.args.human_node_rnn_size)).cuda()
            c_edges = Variable(torch.zeros(numNodes * numNodes, net.args.human_human_edge_rnn_size)).cuda()

            # TODO Not keeping track of nodes leaving the frame (or new nodes entering the frame, which I don't think we can do anyway)
            # Forward prop
            outputs, h_nodes, h_edges, c_nodes, c_edges, attn_w = net(ret_nodes[tstep+1-args.obs_length:,:,:], ret_edges[tstep+1-args.obs_length:,:,:],
                                                                      nodesPresent[tstep+1-args.obs_length:], 
                                                                      edgesPresent[tstep+1-args.obs_length:], h_nodes, h_edges, c_nodes, c_edges)

            ret_nodes[tstep + 1, :, 0] = outputs[-1,:,0]

            # Compute edges
            # TODO Currently, assuming edges from the last observed time-step will stay for the entire prediction length
            ret_edges[tstep + 1, :, :] = fake_edge(ret_nodes.data, tstep, edgesPresent[args.obs_length-1])


            #print(f'{tstep}:{ret_nodes[tstep:,100,:]}\n')

            # Store computed attention weights
            ret_attn.append(attn_w[-1])

    return ret_nodes[:ret_nodes.size()[0]-1], ret_attn


if __name__ == '__main__':
    main()
