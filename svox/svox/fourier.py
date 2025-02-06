import os.path as osp
import torch
import numpy as np
import math
from torch import nn, autograd
from svox.helpers import DataFormat



class FourierConverter():
    def __init__(self, n1, n2, sh_dim, timesteps, device= "cpu"):
        self.n1 = n1
        self.n2 = n2
        self.sh_dim = sh_dim
        self.l_max = int(math.sqrt(sh_dim)) -1
        self.T = timesteps
        self.DFT_n1, self.DFT_n2 = self.getDFTmatrix(device)
        self.IDFT_n1, self.IDFT_n2 = self.getIDFTmatrix(device)
        assert (self.l_max +1)**2 == self.sh_dim, "SH_dim not square: ("+str(self.l_max)+ "+"+str(1)+")**"+str(2)+")" +" != "+ str(self.sh_dim)
    
    @classmethod
    def from_DataFormat(self,data_format,timesteps, device= "cpu"):
        if data_format.format != DataFormat.FC and data_format.format != DataFormat.LFC:
            print("Wrong data format, needs to be FC or LFC")
        return FourierConverter(data_format.fc_dim1,data_format.fc_dim2,data_format.basis_dim, timesteps, device)

    def __repr__(self):
        return (f"svox.FourierConverter(n1={self.n1}, n2={self.n2}, sh_dim={self.sh_dim}, l_max={self.l_max}, timesteps={self.T})");

    def getDFTmatrix(self, device = "cpu"):
        DFT_n1 = torch.zeros((self.n1,self.T), device=device)
        for i in range(self.n1):
            for t in range(self.T):
                DFT_n1[i,t] = self.DFT(i,t)
        DFT_n2 = torch.zeros((self.n2,self.T), device=device)
        for i in range(self.n2):
            for t in range(self.T):
                DFT_n2[i,t] = self.DFT(i,t)
        return DFT_n1, DFT_n2

    def getIDFTmatrix(self, device = "cpu"):
        IDFT_n1 = torch.zeros((self.T,self.n1), device=device)
        for i in range(self.n1):
            for t in range(self.T):
                IDFT_n1[t,i] = self.IDFT(i,t)
        IDFT_n2 = torch.zeros((self.T,self.n2), device=device)
        for i in range(self.n2):
            for t in range(self.T):
                IDFT_n2[t,i] = self.IDFT(i,t)
        return IDFT_n1, IDFT_n2
    
    def IDFT(self,i,t):
        if i%2 == 0:
            fct = (i * np.pi)/self.T
            out = np.cos(t*fct)
        else:
            fct = ((i+1) * np.pi)/self.T
            out = np.sin(t*fct)
        return out

    def DFT(self,i,t):
        if i%2 == 0:
            fct = (i * np.pi)/self.T
            out = np.cos((t)*fct)
        else:
            fct = ((i+1) * np.pi)/self.T
            out = np.sin((t)*fct)
        return out/self.T

    def fourier2sh(self, fc, t):
        fc_2, fc_1 = torch.split(fc, [3*self.sh_dim*self.n2, self.n1], dim=1)
        fc_1 = torch.transpose(torch.reshape(fc_1,(fc.shape[0],-1, self.n1)),1,2)
        fc_2 = torch.transpose(torch.reshape(fc_2,(fc.shape[0],-1, self.n2)),1,2)
        sh_1 = torch.matmul(torch.reshape(self.IDFT_n1[t],(1,-1)).unsqueeze(0),fc_1).squeeze(1)
        sh_2 = torch.matmul(torch.reshape(self.IDFT_n2[t],(1,-1)).unsqueeze(0),fc_2).squeeze(1)
        sh = torch.cat([sh_2,sh_1], dim=1)
        return sh

    def fourier2sh_float(self, fc, t): # for inbetween timesteps
        IDFT_n1 = torch.zeros((1,self.n1), device=fc.device)
        for i in range(self.n1):
            IDFT_n1[0,i] = self.IDFT(i,t)
        IDFT_n2 = torch.zeros((1,self.n2), device=fc.device)
        for i in range(self.n2):
            IDFT_n2[0,i] = self.IDFT(i,t)
        fc_2, fc_1 = torch.split(fc, [3*self.sh_dim*self.n2, self.n1], dim=1)
        fc_1 = torch.transpose(torch.reshape(fc_1,(fc.shape[0],-1, self.n1)),1,2)
        fc_2 = torch.transpose(torch.reshape(fc_2,(fc.shape[0],-1, self.n2)),1,2)
        sh_1 = torch.matmul(torch.reshape(IDFT_n1,(1,-1)).unsqueeze(0),fc_1).squeeze(1)
        sh_2 = torch.matmul(torch.reshape(IDFT_n2,(1,-1)).unsqueeze(0),fc_2).squeeze(1)
        sh = torch.cat([sh_2,sh_1], dim=1)
        return sh

    def sh2fourier(self, sh):
        fc_1 = torch.matmul(self.DFT_n1.unsqueeze(0), sh[:,:,-1].unsqueeze(2))
        fc_2 = torch.matmul(self.DFT_n2.unsqueeze(0), sh[:,:,:-1])
        fc_1 = torch.reshape(torch.transpose(fc_1,1,2),(sh.shape[0],-1))
        fc_2 = torch.reshape(torch.transpose(fc_2,1,2),(sh.shape[0],-1))
        fc = torch.cat([fc_2, fc_1], dim=1)
        return fc