import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils import generate_original_PE, generate_regular_PE, DenseInterpolation


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1, dropout=0, batch=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 1

        self.vars = nn.ParameterList()

        kernel = nn.Parameter(torch.ones(input_size, 4*hidden_size, device="cuda")) #nn.Linear(d_model, q*h)
        torch.nn.init.kaiming_normal_(kernel)
        self.vars.append(kernel)

        recurrent_kernelLSTM = nn.Parameter(torch.ones(4*hidden_size, hidden_size, device="cuda")) #nn.Linear(d_model, q*h)
        torch.nn.init.kaiming_normal_(recurrent_kernelLSTM)
        self.vars.append(recurrent_kernelLSTM)

        biasLSTM = nn.Parameter(torch.zeros(4*hidden_size, device="cuda"))
        self.vars.append(biasLSTM)

        Woutput = nn.Parameter(torch.ones(output_size, hidden_size, device="cuda")) #nn.Linear(d_model, q*h)
        torch.nn.init.kaiming_normal_(Woutput)
        self.vars.append(Woutput)

        boutput = nn.Parameter(torch.zeros(output_size, device="cuda"))
        self.vars.append(boutput)

        self._dropout = nn.Dropout(p=dropout)
        self._sigmoid = nn.Sigmoid()
        self._tanh = nn.Tanh()


    def lstmcell(self, x_t, h_tm1, c_tm1, kernel, recurrent_kernel, bias,Woutput=None,boutput=None, train=False, keep_prob_recurrent=None):
        kernel_out = torch.matmul(x_t, kernel) #tf.matmul(x_t,kernel)

        recurrent_kernel_out = F.linear(h_tm1,recurrent_kernel, bias) #tf.matmul(h_tm1,recurrent_kernel) + bias

        # if keep_prob_recurrent is not None:
            # kernel_out = tf.nn.dropout(kernel_out,keep_prob_recurrent)
            # recurrent_kernel_out = tf.nn.dropout(recurrent_kernel_out,keep_prob_recurrent)

        x0,x1,x2,x3 = torch.split(kernel_out, kernel_out.size()[1]//4, dim=1) #tf.split(kernel_out,num_or_size_splits=4,axis=1)
        r0,r1,r2,r3 = torch.split(recurrent_kernel_out, recurrent_kernel_out.size()[1]//4, dim=1) #tf.split(recurrent_kernel_out,num_or_size_splits=4,axis=1)

        f = self._sigmoid(x0.add(r0))
        i = self._sigmoid(x1.add(r1))
        c_prime = self._tanh(x2.add(r2))
        c_t = f*c_tm1 + i * c_prime
        o = self._sigmoid(x3.add(r3))
        h_t = o*self._tanh(c_t)

        if Woutput is not None:
            output_t = F.linear(h_t, Woutput, boutput) #tf.matmul(h_t,Woutput) + boutput
        else:
            output_t = h_t
        if train:
            output_t = self._dropout(output_t)
        return output_t,h_t,c_t

    def forward(self, input_seq, vars=None, train=False): #vars=None,
        if vars == None:
            vars = self.vars
        K = input_seq.shape[1]
        # input_seq = torch.Tensor(input_seq)
        # if vars == None:
        device = torch.device("cuda")
        ht = torch.zeros(input_seq.size(0), self.hidden_size).to(device)
        ct = torch.zeros(input_seq.size(0), self.hidden_size).to(device) #self.num_layers, input_seq.size(0),

        input_seq = input_seq.transpose(0, 1)
        for i in range(K):
            lstm_out, ht, ct = self.lstmcell(input_seq[i], ht, ct, vars[0], vars[1], vars[2], vars[3], vars[4], train=train)

        predictions = lstm_out.view(-1)

        return predictions, ht
    

class PhasedLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, leak=0.001, ratio_on=0.1, period_init_min=1.0, period_init_max=1000.0):
        super(PhasedLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.leak = leak
        self.ratio_on = ratio_on

        # self.cell = nn.LSTMCell(hidden_size, hidden_size)
        
        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        log_period = torch.empty(hidden_size).uniform_(math.log(period_init_min), math.log(period_init_max))
        self.tau = nn.Parameter(torch.exp(log_period))
        self.phase = nn.Parameter(torch.rand(hidden_size) * self.tau)

    def forward(self, input, hidden, time):
        h_prev, c_prev = hidden

        gates = self.input_weights(input) + self.hidden_weights(h_prev)
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, 1)
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        o = torch.sigmoid(o_gate)
        g = torch.tanh(g_gate)

        phi = ((time - self.phase) % self.tau) / self.tau
        k = torch.where(phi < self.ratio_on, 1.0, self.leak * phi)

        c = f * c_prev + i * g * k.unsqueeze(0)
        h = o * torch.tanh(c) * k.unsqueeze(0)

        return h, c
