from sympy import * 
import torch 
from scipy.stats import norm 
import math 
import numpy as np

symbolic = True

class Sampler():
    def __init__(self, sigma, shape= (1000,3,32,32), total=100000):
        k = max(1,int(6*sigma))
        x =  symbols('x')
        thresholds = [0] *(2*255*(k+1)+1)

        for idx, i in enumerate(range(-255*(k+1),  255*(k+1))):
            if(symbolic):
                expr = integrate(exp(-x*x/(2*sigma*sigma))/ sqrt(2*pi*sigma*sigma), (x,-oo,  Rational(2*i+1, 510)))*2**32
                expr = expr.subs(x, Rational(2*i+1, 510))
                thresholds[idx] = math.ceil(N(expr, 60))
            else:
                thresholds[idx] = math.ceil(norm.cdf((2*i+1)/510, scale=sigma)*2**32)
        thresholds[-1] = 2**32


        self.thresholds = thresholds

        for idx in range(total//shape[0]+1):
            tmp = torch.randint(0, 2**32, shape)
            noise = torch.zeros(*shape)
            for i in thresholds:
                noise  += tmp < i
            noise = (np.array(noise)-255*(k+1)-1)/255
            np.save('A' + str(idx) + '.npy', noise)
        
        self.state = 0
        self.tot = total//shape[0]+1
        self.k = k


    def sample_noise(self, x):
        noise = torch.zeros(x.shape, device='cuda')
        tmp = torch.randint(0,2**32, x.shape).to('cuda')
        for i in self.thresholds:
            noise += tmp < i 
        return (x+noise).clip(-self.k, self.k+1)



class Sampler2():
    def __init__(self, sigma, shape = None, total=None):
        k = max(1,int(6*sigma))
        x =  symbols('x')
        thresholds = [0] *(2*255*(k+1)+1)
        for idx, i in enumerate(range(-255*(k+1),  255*(k+1))):
            symbolic = 0
            if(i == 0):
                print(idx)
            if(symbolic):
                expr = integrate(exp(-x*x/(2*sigma*sigma))/ sqrt(2*pi*sigma*sigma), (x,-oo,  Rational(2*i+1, 510)))*2**32
                expr = expr.subs(x, Rational(2*i+1, 510))
                thresholds[idx] = math.ceil(N(expr, 60))
            else:
                thresholds[idx] = math.ceil(norm.cdf((2*i+1)/510, scale=sigma)*2**32)
        thresholds[-1] = 2**32


        self.thresholds = thresholds

        self.weights = torch.tensor(thresholds[0:1] + [i-j for i,j in zip(thresholds[1:], thresholds[:-1])], dtype=torch.float)
        self.weights
        self.k = k


    def sample_noise(self, x):
        siz = np.prod(x.shape)
        noise = (torch.multinomial(self.weights, replacement=True,num_samples=siz).reshape(x.shape)-255*(self.k+1))/255.
        return torch.clip(x+noise, -self.k, self.k+1)

class Sampler3():
    def __init__(self, sigma, shape= (1000,3,32,32), total=100000):
        k = max(1,int(6*sigma))
        x =  symbols('x')
        thresholds = [0] *(2*255*(k+1)+1)

        for idx, i in enumerate(range(-255*(k+1),  255*(k+1))):
            if(symbolic):
                expr = integrate(exp(-x*x/(2*sigma*sigma))/ sqrt(2*pi*sigma*sigma), (x,-oo,  Rational(2*i+1, 510)))*2**32
                expr = expr.subs(x, Rational(2*i+1, 510))
                thresholds[idx] = math.ceil(N(expr, 60))
            else:
                thresholds[idx] = math.ceil(norm.cdf((2*i+1)/510, scale=sigma)*2**32)
        thresholds[-1] = 2**32


        self.thresholds = thresholds

        for idx in range(total//shape[0]+1):
            tmp = torch.randint(0, 2**32, shape)
            noise = torch.zeros(*shape)
            for i in thresholds:
                noise  += tmp < i
            noise = (np.array(noise)-255*(k+1)-1)/255
            np.save('A' + str(idx) + '.npy', noise)
        
        self.state = 0
        self.tot = total//shape[0]+1
        self.k = k


    def sample_noise(self, x):
        noise = np.load('A'+str(self.state) + '.npy')[:x.shape[0]]
        noise = torch.tensor(noise, device=x.device)
        self.state += 1
        if(self.state >= self.tot):
            self.state = 0

        return torch.clip(x+noise, -self.k, self.k+1)
