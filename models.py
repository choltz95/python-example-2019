import torch
import gpytorch
import torch.nn as nn
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import pandas as pd 
import numpy as np

class GPCNNLSTM(nn.Module):
    def __init__(self, input_dim, lstm_hidden=32):
        super(GPCNNLSTM, self).__init__()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.predictor_model = CRNN(input_dim[1], lstm_hidden, n_class = 2, leakyRelu=True).cuda()
        #self.interpolation_model = MTGPInterpolationModel(input_dim, likelihood).cuda()
        #params = [
        #            {'params': self.predictor_model.parameters()},
        #            {'params': self.interpolation_model.parameters()}
        #         ]
        self.optimizer = optim.AdamW(self.predictor_model.parameters(), lr = 0.0001)
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor([0.15,0.85]).cuda(),reduction='mean')
        #self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.output_transform = nn.Softmax(dim=1)
        #self.loss = nn.CrossEntropyLoss(reduction='mean')

    def update(self, input_data, labels):
        #n_batches = int(labels.shape[0]/self.batch_size) 
        batch_size = len(input_data)
        losses = []
        loss = 0.0
        for j in range(batch_size):
            vital_features, lab_features, baseline_features = input_data[j]
            #with gpytorch.settings.detach_test_caches(False):
            #    self.interpolation_model.train()
            #    self.interpolation_model.likelihood.train()
            #    self.interpolation_model.eval()
            #    gp_output = self.interpolation_model(vital_features)

            self.predictor_model.train()
            self.optimizer.zero_grad()

            #vital_features = gp_output.rsample(torch.Size([10]))
            #f_samples = f_samples.transpose(-2, -1)
            #sample_mean = f_samples.mean(0).squeeze(-1)  # Average over GP sample dimension

            #print(sample_mean.shape)

            vital_features = vital_features.permute(0, 2, 1) # b d t
            #vital_features = sample_mean.permute(0, 2, 1) # b d t

            output = self.predictor_model(vital_features).squeeze(0)
            output = self.output_transform(output)

            loss += self.loss(output, labels[j].squeeze())
            #output = self.predictor_model(self.interpolation_model(vital_features))

        loss = loss / batch_size
        losses.append(loss.cpu().item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor_model.parameters(), 0.25)

        """
        task_losses = torch.stack([sepsis_head, duration_head])
        weighted_losses = self.weights * task_losses
        total_weighted_loss = weighted_losses.sum()
        # compute and retain gradients
        total_weighted_loss.backward(retain_graph=True)
        # zero the w_i(t) gradients since we want to update the weights using gradnorm loss
        self.weights.grad = 0.0 * self.weights.grad

        W = list(self.model.mlp_list[-1].parameters())
        norms = []

        for w_i, L_i in zip(self.weights, task_losses):
                gLgW = torch.autograd.grad(L_i, W, retain_graph = True)
                norms.append(torch.norm(w_i * gLgW[0]))

        norms = torch.stack(norms)

        if t ==0:
                self.initial_losses = task_losses.detach()

        # compute the constant term without accumulating gradients
        # as it should stay constant during back-propagation
        with torch.no_grad():
                # loss ratios \curl{L}(t)
                loss_ratios = task_losses / self.initial_losses
                # inverse training rate r(t)
                inverse_train_rates = loss_ratios / loss_ratios.mean()
                constant_term = norms.mean() * (inverse_train_rates ** self.alpha)

        grad_norm_loss = (norms - constant_term).abs().sum()
        self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]
        """

        self.optimizer.step()
        del output
        torch.cuda.empty_cache()
        del loss

        return np.mean(losses)

    def predict(self, input_data):
        self.predictor_model.eval()
        #self.interpolation_model.eval()
        #self.interpolation_model.likelihood.eval()

        vital_features, lab_features, baseline_features = input_data
        #gp_output = self.interpolation_model(vital_features)
        #vital_features = gp_output.rsample(torch.Size([10]))
        #vital_features = gp_output.rsample(torch.Size([10]))
        #f_samples = f_samples.transpose(-2, -1)
        #sample_mean = f_samples.mean(0).squeeze(-1)  # Average over GP sample dimension
        #vital_features = sample_mean.permute(0, 2, 1) # b d t
        #vital_features = vital_features.transpose(-2,-1)
        return self.output_transform(self.predictor_model(vital_features.permute(0, 2, 1)).squeeze(0))

class MTGPInterpolationModel(gpytorch.models.ExactGP):
    def __init__(self, likelihood):
        super(MTGPInterpolationModel, self).__init__(likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=7, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar) 


class LSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(LSTM, self).__init__()
        self.rnn = nn.GRU(nIn, nHidden, bidirectional=False)
        self.embedding = nn.Linear(nHidden, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, input_dim, lstm_hidden, n_class, leakyRelu=True):
        super(CRNN, self).__init__()

        kernel_size = [3, 3, 3, 3, 3, 3, 3]
        pad_size = [1, 1, 1, 1, 1, 1, 0]
        shift_size = [1, 1, 1, 1, 1, 1, 1]
        in_size = [16, 32, 32, 64, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False, dropout=False):
            nIn = input_dim if i == 0 else in_size[i - 1]
            nOut = in_size[i]
            cnn.add_module('conv{0}'.format(i),
                       nn.Conv1d(nIn, nOut, kernel_size[i], shift_size[i], pad_size[i]))
            if dropout:
                cnn.add_module('dropout{0}'.format(i), nn.Dropout(0.1))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm1d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        convRelu(1)
        convRelu(2)
        convRelu(3, True)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            LSTM(64, lstm_hidden, n_class))

        #self.register_backward_hook(self.backward_hook)

    def forward(self, input):
        # conv features
        conv = self.cnn(input) 
        b, d, t = conv.size()
        assert t == input.shape[-1], "t must be the same as input"
        conv = conv.permute(0, 2, 1)  # [t, b, c]

        # rnn features
        output = self.rnn(conv)
        return output

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero