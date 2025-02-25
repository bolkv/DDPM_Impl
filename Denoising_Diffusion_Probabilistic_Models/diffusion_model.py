import torch
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
from keras.datasets.mnist import load_data

(trainX, trainy), (testX, testy) = load_data()
trainX = np.float32(trainX) / 255.
testX = np.float32(testX) / 255.

def sample_batch(batch_size, device):
    indices = torch.randperm(trainX.shape[0])[:batch_size]
    data = torch.from_numpy(trainX[indices]).unsqueeze(1).to(device)
    return torch.nn.functional.interpolate(data, 32)

class DiffusionModel(nn.Module):

    def __init__(self, T, model, device):
        super().__init__()
        self.T = T
        self.function_approximator = model.to(device)
        self.device = device

        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = (1 - self.beta)
        self.alpha_bar = torch.cumprod(self.alpha, dim = 0)

    def training_step(self, batch_size, optimizer):
        '''
        Algorithm 1 in Denoising Diffusion Probability Models
        '''
        x0 = sample_batch(batch_size, self.device)

        t = torch.randint(1, self.T + 1, (batch_size, ), device = self.device, dtype = torch.long)
        eps = torch.randn_like(x0, device=self.device)

        #Take one gradient descent step
        alpha_bar_t = self.alpha_bar[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        eps_predicted = self.function_approximator(torch.sqrt(alpha_bar_t)*x0 + 
                                                   torch.sqrt(1-alpha_bar_t)*eps,t-1)
        loss = nn.functional.mse_loss(eps, eps_predicted)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sampling(self, n_samples=1, image_channel=1, image_size=(32, 32), use_tqdm=True):
        
        xT = torch.randn((n_samples, image_channel, image_size[0], image_size[1]), device=self.device)
        x = xT
        progress_bar = tqdm if use_tqdm else lambda x : x
        for t in progress_bar(range(self.T, 0, -1)):
            if(t == 0):
                z = torch.zeros_like(x, device = self.device)
            else:
                z = torch.randn_like(x, device = self.device)

            t = torch.ones(n_samples, dtype=torch.long, device=self.device) * t
                
            alpha_t = self.alpha[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_t = self.alpha_bar[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            beta_t = self.beta[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            eps_theta = self.function_approximator(x,t-1)
            
            mean = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(
                1 - alpha_bar_t)) * eps_theta)
            sigma = torch.sqrt(beta_t)
            x =  mean + sigma * z

        return x

