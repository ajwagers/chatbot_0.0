# model.py
import torch
import torch.nn as nn

#This is a Feed Forward Neural Network (FNN) with 2 hidden layers
'''
"A feed-forward neural network is a classification algorithm that consists 
of a large number of perceptrons, organized in layers & each unit in the 
layer is connected with all the units or neurons present in the previous 
layer. These connections are not all equal and can differ in strengths or 
weights. The weights on these connections cipher the knowledge of the network."
from:  https://analyticsindiamag.com/guide-to-feed-forward-network-using-pytorch-with-mnist-dataset/
see also:  https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/
'''


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
