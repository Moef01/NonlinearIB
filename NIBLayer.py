# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 21:24:46 2020

@author: Morten Østergaard Nielsen
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


#%% Utility functions
def softplusinverse(x):
    '''
    Stable implementation of the inverse softplus function.
    Softplus:
        s(x) = log(1+exp(x))
    
    Inverse Softplus:
         s^-1(x) = log(1-exp(-x)) + x

    Parameters
    ----------
    x : torch.Tensor
        Function variable.

    Returns
    -------
    torch.Tensor
        The value of the inverse softplus function.
    '''
    return torch.log(1. - torch.exp(-x)) + x

# def matrix_softplus(X):
    
#     U,S,V = X.svd()
  
#     D = torch.exp(S)
#     A = (U*D).matmul(V.T)
#     A = 1. + A
#     U,S,V = A.svd()
#     D = torch.log(S)

#     return (U*D).matmul(V.T)

# def matrix_inverse_softplus(X):
    
#     U,S,V = X.svd()
    
#     D = torch.exp(-S)
#     A = (U*D).matmul(V.T)
#     A = 1. - A
#     U,S,V = A.svd()
#     D = torch.log(S)

#     return (U*D).matmul(V.T) + X

def pairwise_distance(x, p = 2):
    '''
    This function computes the pairwise lp-norm of the samples in the x-tensor.
    The first dim/axis of the tensor is assumed to be the sample axis.
    
    Parameters
    ----------
    x : torch.Tensor
        Input variable.
    p : int, optional
        Determines which lp-norm to use. The default is 2.

    Returns
    -------
    dist : torch.Tensor
        Distance matrix of the pairwise distance between samples.

    '''
    #Flatten or expand the tensor dimension as requirred.
    if x.ndim > 2:
        x = x.view(x.size(0), -1)
    elif x.ndim == 1:
        x = x.view(-1,1)
    
    
    dx = x[:, None] - x     #Easy way to do pairwise subtraction
    dist = torch.norm(dx, p = p, dim = -1)  #Compute the lp-norm
    return dist

def nats2bits(x):
    '''
    Converting the unit of the input from nats to bits.

    Parameters
    ----------
    x : torch.Tensor
        Input variable in nats.

    Returns
    -------
    torch.Tensor
        Input variable in bits.

    '''
    return x/torch.log(2.*torch.ones_like(x))

def bits2nats(x):
    '''
    Converting the unit of the input from bits to nats.

    Parameters
    ----------
    x : torch.Tensor
        Input variable in bits.

    Returns
    -------
    torch.Tensor
        Input variable in nats.

    '''
    return x*torch.log(2.*torch.ones_like(x))

#%%Nonlinear Information Bottleneck Layer
class NIBLayer(nn.Module):
    r'''
    This is a PyTorch implementation of the Nonlinear Information Bottleneck 
    (NIB) by Kolchinsky et al.[doi:10.3390/e21121181]. This implementation is 
    inspired by the authors' Tensorflow/Keras implementation, where the NIB 
    have been coded as a Keras layer. 
    Similarly, the NIB have been implemented as a pair of PyTorch layer and 
    loss function.
    
    The following is the NIB layer, which adds gaussian noise to the encoded
    features. OBS! This layer can be use seperately in any model but must be
    included in order to use the NIBLoss!
    '''
    def __init__(self, noisevar=1., train_var = True, bits = True):
        '''
        Parameters
        ----------
        noisevar : float, optional
            The variance of the additive i.i.d. Gaussian noise. 
            The default is 1.
        train_var : bool, optional
            Boolean variable. Allows the noise variance to be learned during 
            training. The default is True.
        bits : bool, optional
            Boolean variable. Specify if the unit is nats (False) or bits (True) 
            The default is True.

        Returns
        -------
        None.

        '''
        super(NIBLayer, self).__init__()
        
        #Transform the noise variance to the inverse softplus domain.
        #This guarantees that the noise variance stays positive when learned during training.
        init_phi = softplusinverse(torch.tensor(noisevar))
        
        #Store the variables
        self.phi = nn.Parameter(init_phi, requires_grad = train_var)
        self.init_noisevar = F.softplus(self.phi)
        self.bits = bits
        
    def KDE_IXT_estimator(self, input):
        '''
        This is a upper bound approximation of the encoding mutual information
        as proposed by Kolchinsky et al.[doi:10.3390/e21121181]. 
        Based on a kernel density estimator, the encoding mutual information 
        I(X;T) is estimated. The estimate is stored as a parameter of the layer.
        
        Parameters
        ----------
        input : torch.Tensor
            The layer input(s) from the previous layer in the model.

        Returns
        -------
        None.

        '''
        n_batch, _ = input.shape        #Batch samples
        var = F.softplus(self.phi)      #Get the variance
        
        # Compute normalisation constant
        c = np.log(n_batch)
        
        #Compute mean log-sum-exp
        dist = pairwise_distance(input)**2
        kde_contribution = torch.mean(torch.logsumexp(-0.5*dist/var, dim = 1))
        IXT = c - kde_contribution
        
        if self.bits:   #Convert to bits
            IXT = nats2bits(IXT)
         
        #Add I(X;T) to the layer as a non-trainable parameter    
        self.IXT = nn.Parameter(IXT, requires_grad = False)
        
    def forward(self, input, training = True):
        '''
        The forward method of the layer.

        Parameters
        ----------
        input : torch.Tensor
            The layer input(s) from the previous layer in the model.
        training : bool, optional
            Boolean variable. The layer should only add Gaussian noise to the 
            input during training. Set to False when testing. The default is 
            True.

        Returns
        -------
        output : torch.Tensor
            Input features with added Gaussian noise.

        '''
        #Estimate the encoder MI
        self.KDE_IXT_estimator(input)
        
        if training:    #During training
            #Generate standard gaussian noise with the same shape as the input
            noise = torch.randn_like(input)
            #Compute the layer output
            output = input + F.softplus(self.phi) * noise
        else: #During testing
            output = input
        return output
    
    def extra_repr(self):
        '''
        Extra representation of the layer. Similar to other PyTorch layers.

        Returns
        -------
        str
            String with variables of the layer.

        '''
        return 'init_noisevar={}, noisevar={}, phi={:.5f}'.format(
            self.init_noisevar, F.softplus(self.phi).item(), self.phi.item()
        )

#%% Nonlinear Information Bottleneck Loss Function
class NIBLoss(nn.Module):
    r'''
    This is a PyTorch implementation of the Nonlinear Information Bottleneck 
    (NIB) by Kolchinsky et al.[doi:10.3390/e21121181]. This implementation is 
    inspired by the authors' Tensorflow/Keras implementation, where the NIB 
    have been coded as a Keras layer. 
    Similarly, the NIB have been implemented as a pair of PyTorch layer and 
    loss function.
    
    The following is the NIB loss function, which adds gaussian noise to the 
    encoded features. OBS! This loss function cannot be used seperately and 
    must be combined with the NIBLayer!
    '''
    def __init__(self, beta, h_func = "exp", gamma = 1., discrete_output = True, 
                 varY = None, bits = True, **kwargs):
        '''
        Parameters
        ----------
        beta : float
            Regualization parameter of the information bottleneck.
        h_func : str, optional
            Name of the non-linear function to apply on the encoder MI. Avail-
            able functions are; 'exp', 'pow' and 'linear'. The default is "exp".
        gamma : float, optional
            Parameter of the non-linear function, e.g. exp(gamma * x). 
            The default is 1..
        discrete_output : bool, optional
            Boolean variable. If the output of the DNN model is discrete(True), 
            e.g. classification problem, Cross-entropy loss is used to estimate 
            the decoder MI. Otherwise(False), MSE is used instead. 
            The default is True.
        varY : float or None, optional
            The variance of the output. Only necessary for continuous outputs.
            The default is None.
        bits : bool, optional
            Boolean variable. Specify if the unit is nats (False) or bits (True) 
            The default is True.
        **kwargs : 
            Additional keyword arguments are passed to the Cross-entropy or MSE
            loss function used for estimating the deconder MI.

        '''
        super(NIBLoss, self).__init__()
        
        #Store the variables
        self.beta = beta
        self.h_func_param = gamma
        self.bits = bits
        
        #Define the non-linear function.
        if "exp" in h_func.lower():
            self.h_func = lambda r: torch.exp(gamma * r)
            self.h_func_name = "exp"
        elif "pow" in h_func.lower():
            self.h_func = lambda r: r**(1 + gamma)
            self.h_func_name = "pow"
        elif "zero" in h_func.lower():
            self.h_func = lambda r: 0.
            self.h_func_name = "zero"
        else:
            #If the wanted functions is not avaialble an identity function will
            #be applied instead.
            self.h_func = lambda r: r
            self.h_func_name = "linear"
         
        self.discrete_output = discrete_output
        if discrete_output:
            #If the output is discrete the decoder MI is lower bounded by
            #a constant subtracted the cross-entropy.
            self.loss_func = nn.CrossEntropyLoss(**kwargs)
            self.HY = 0.
        else:
            #If the output is continuous the decoder MI is lower bounded by
            #a constant subtracted the MSE.
            assert not(varY is None), "An estimate of the target variance is needed when dealing with continuous outputs. Estimate the variance from the available training set."
            self.loss_func = nn.MSELoss(**kwargs)
            self.HY = 0.5 * (np.log(2.0 * np.pi * np.e) + np.log(varY))
            self.varY = varY
            
    def get_ITY(self, y_pred, y_true):
        '''
        Computes the decoder mutual information I(T;Y) using a lower bound.
        Parameters
        ----------
        y_pred : torch.Tensor
            The output from the DNN model.
        y_true : torch.Tensor
            The corresponding targets.

        Returns
        -------
        ITY : torch.Tensor
            The estimated lower bound on the decoder MI I(T;Y).
        ITY_lower: torch.Tensor
            If the output is discrete, this is the same as ITY.
            Otherwise, a different lower bound that is easier for optimization.

        '''
        if self.discrete_output:
            self.HY = np.log(y_pred.size(1))    #Maximum Entropy for discrete stocastic variables
            ITY = self.HY - self.loss_func(y_pred, y_true)  #Lower bound
            if self.bits:
                ITY = nats2bits(ITY)
            return ITY, ITY
        else:
            MSE = self.loss_func(y_pred, y_true)
            ITY = 0.5 * (torch.log(self.varY) - torch.log(MSE)) 
            ITY_lower = self.HY - MSE
            if self.bits:
                ITY = nats2bits(ITY)
                ITY_lower = nats2bits(ITY_lower)
            return ITY, ITY_lower
    
    def forward(self, input, target, IXT_upper):
        '''
        The forward method of the loss function.
        Computes the Nonlinear Information Bottleneck loss:
            
            L = -I(T;Y) + beta * h[I(X;T)]

        Parameters
        ----------
        input : torch.Tensor
            The predicted target(s) from the output of the DNN model.
        target : torch.Tensor
            The true target(s).
        IXT_upper : torch.Tensor
            The upper bound of the encoder MI I(X;T). This is obtained from the
            NIBLayer.
        

        Returns
        -------
        ITY : torch.Tensor
            The estimated lower bound on the decoder MI I(T;Y).
        loss : torch.Tensor
            The loss 

        '''
        ITY, ITY_lower = self.get_ITY(input, target)
        loss = -1.0 * (ITY_lower - self.beta * self.h_func(IXT_upper))
        return ITY, loss
            
            