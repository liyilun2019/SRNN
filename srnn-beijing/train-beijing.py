'''
Train script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 29th March 2017
'''

import argparse
import os
import pickle
import time

import torch
from torch.autograd import Variable
# torch.cuda.set_device(1)

from model import SRNN
from my_graph import My_Graph
from my_loader import My_DataPointer
from my_utils import getL2Loss


def main():
    parser = argparse.ArgumentParser()

    # RNN size
    parser.add_argument('--human_node_rnn_size', type=int, default=128,
                        help='Size of Human Node RNN hidden state')
    parser.add_argument('--human_human_edge_rnn_size', type=int, default=256,
                        help='Size of Human Human Edge RNN hidden state')

    # Input and output size
    parser.add_argument('--human_node_input_size', type=int, default=1,
                        help='Dimension of the node features')
    parser.add_argument('--human_human_edge_input_size', type=int, default=2,
                        help='Dimension of the edge features')
    parser.add_argument('--human_node_output_size', type=int, default=1,
                        help='Dimension of the node output')

    # Embedding size
    parser.add_argument('--human_node_embedding_size', type=int, default=64,
                        help='Embedding size of node features')
    parser.add_argument('--human_human_edge_embedding_size', type=int, default=64,
                        help='Embedding size of edge features')

    # Attention vector dimension
    parser.add_argument('--attention_size', type=int, default=64,
                        help='Attention size')

    # Sequence length
    parser.add_argument('--seq_length', type=int, default=16,
                        help='Sequence length')
    parser.add_argument('--pred_length', type=int, default=1,
                        help='Predicted sequence length')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')

    # Number of epochs
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')

    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.00005,
                        help='L2 regularization parameter')

    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='decay rate for the optimizer')

    # Dropout rate
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout probability')

    # The leave out dataset
    parser.add_argument('--leaveDataset', type=int, default=3,
                        help='The dataset index to be left out in training')

    parser.add_argument('--load', type=int, default=-1,
                    help='<0: not load, other: epoch to load')

    args = parser.parse_args()

    train(args)


def load_net(save_directory,epoch,net):
    checkpoint_path = os.path.join(save_directory, 'srnn_model_'+str(epoch)+'.tar')

    if os.path.isfile(checkpoint_path):
        print('Loading checkpoint')
        checkpoint = torch.load(checkpoint_path)
        # model_iteration = checkpoint['iteration']
        model_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        print('Loaded checkpoint at {}'.format(model_epoch))


def train(args):
    datasets = [i for i in range(5)]
    # Remove the leave out dataset from the datasets
    datasets.remove(args.leaveDataset)
    # datasets = [0]
    # args.leaveDataset = 0
    # 
    # Construct the ST-graph object
    filedir='../beijing/'
    grp = My_Graph(args.seq_length+1,filedir+'W.pk',filedir+'traffic_data.csv')

    my_datapt=My_DataPointer(grp.getFrameNum(),args.seq_length+1,args.batch_size)

    # Log directory
    log_directory = '../log-beijing-16/'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    # Logging file
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')

    # Save directory
    save_directory = '../save-beijing-16/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # Open the configuration file
    with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, 'srnn_model_'+str(x)+'.tar')

    # Initialize net
    net = SRNN(args)
    net.cuda()

    if(args.load>=0):
        load_net(save_directory,args.load,net)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, momentum=0.0001, centered=True)

    learning_rate = args.learning_rate
    print('Training begin')
    best_val_loss = 100
    best_epoch = 0

    print(f"train_batches={my_datapt.num_batches()}")
    print(f"val_batches={my_datapt.valid_num_batches()}")


    # Training
    for epoch in range(args.num_epochs):
        log_output=open(log_directory+'log_output.py',"w") 
        log_output.write("pairs=[\n")

        loss_epoch = 0
        my_datapt.train_reset()

        # For each batch
        for batch in range(my_datapt.num_batches()):
            start = time.time()
            # Loss for this batch
            loss_batch = 0



            # For each sequence in the batch
            for st in my_datapt.get_batch():
                # Construct the graph for the current sequence
                t1=time.time()
                nodes, edges, nodesPresent, edgesPresent = grp.getSequence(st)

                # Convert to cuda variables
                nodes = Variable(torch.from_numpy(nodes).float()).cuda()
                edges = Variable(torch.from_numpy(edges).float()).cuda()

                # Define hidden states
                numNodes = nodes.size()[1]
                hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()

                cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()

                # Zero out the gradients
                net.zero_grad()
                optimizer.zero_grad()
#                print(f"grp time = {time.time()-t1}")
                # Forward prop
                outputs, _, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1], hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs)
                log_output.write(f"[{outputs[-1,:,0].data.cpu().numpy().tolist()},{nodes[-1,:,0].data.cpu().numpy().tolist()}],\n")
#                print(f"forward time = {time.time()-t1}")
                # Compute loss
                loss = getL2Loss(outputs, nodes[1:], nodesPresent[1:], args.pred_length)
                print(f"start={st},loss={loss}")
#                print(f"loss time = {time.time()-t1}")
                loss_batch += loss.data

                # Compute gradients
                loss.backward()
#                print(f"backward time = {time.time()-t1}")
                # Clip gradients
                torch.nn.utils.clip_grad_norm(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()
#                print(f"step time = {time.time()-t1}")

            end = time.time()
            loss_batch = loss_batch / args.batch_size
            loss_epoch += loss_batch

            print(
                '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * my_datapt.num_batches() + batch,
                                                                                    args.num_epochs * my_datapt.num_batches(),
                                                                                    epoch,
                                                                                    loss_batch, end - start))

        # Compute loss for the entire epoch
        loss_epoch /= my_datapt.num_batches()
        # Log it
        log_file_curve.write(str(epoch)+','+str(loss_epoch)+',')

        log_output.write("]\n")


        # Save the model after each epoch
        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))

        # Validation
        my_datapt.val_reset()
        loss_epoch = 0

        with torch.no_grad():
            for batch in range(my_datapt.valid_num_batches()):

                # Loss for this batch
                loss_batch = 0

                for start in my_datapt.get_batch_val():

                    nodes, edges, nodesPresent, edgesPresent = grp.getSequence(start)

                    # Convert to cuda variables
                    nodes = Variable(torch.from_numpy(nodes).float()).cuda()
                    edges = Variable(torch.from_numpy(edges).float()).cuda()

                    # Define hidden states
                    numNodes = nodes.size()[1]
                    hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                    hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()
                    cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
                    cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()

                    outputs, _, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1],
                                                 hidden_states_node_RNNs, hidden_states_edge_RNNs,
                                                 cell_states_node_RNNs, cell_states_edge_RNNs)

                    # Compute loss
                    loss = getL2Loss(outputs, nodes[1:], nodesPresent[1:], args.pred_length)

                    loss_batch += loss.data


                loss_batch = loss_batch / args.batch_size
                loss_epoch += loss_batch

            loss_epoch = loss_epoch / my_datapt.valid_num_batches()

        # Update best validation loss until now
        if loss_epoch < best_val_loss:
            best_val_loss = loss_epoch
            best_epoch = epoch

        # Record best epoch and best validation loss
        print('(epoch {}), valid_loss = {:.3f}'.format(epoch, loss_epoch))
        print('Best epoch {}, Best validation loss {}'.format(best_epoch, best_val_loss))
        # Log it
        log_file_curve.write(str(loss_epoch)+'\n')


    # Record the best epoch and best validation loss overall
    print('Best epoch {}, Best validation loss {}'.format(best_epoch, best_val_loss))
    # Log it
    log_file.write(str(best_epoch)+','+str(best_val_loss))

    # Close logging files
    log_file.close()
    log_file_curve.close()


if __name__ == '__main__':
    main()
