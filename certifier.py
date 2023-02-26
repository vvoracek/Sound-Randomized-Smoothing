import torch 
import math 
import torchvision
from scipy.stats import norm 
from collections import defaultdict, Counter
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm 
import warnings 



class Certifier():
    def __init__(self, sigma, bits_precision=60, device=None, exact = True, symbolic=True):
        """
        :param sigma: standard deviation of smoothed distribution
        :param bits_precision: how much bits to use for random number generation
        :param device: device 
        :param exact: if False, then the certification is unsound.
        :param symbolic: if False, the smoothing distribution is computed with finite precision.
        """
        if(device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device 

        self.k = max(1,int(6*sigma))
        self.sigma = sigma
        self.Q = 2**bits_precision
        self.exact = exact 
        self.symbolic = symbolic

        if(exact):
            self.set_thresholds()


    def get_noise_batch(self, shape):
        with torch.no_grad():
            if(self.exact):
                tmp = torch.randint(0,self.Q, shape, device=self.device)     
                if(tmp.max() >= self.Q-self.increments[-1]-1):
                    return 0, True    
                noise = torch.zeros(shape, device=self.device)
                for idx, i in enumerate(self.thresholds):
                    noise += tmp < i -self.increments[idx]
                return noise/255 -self.k-1, False
            else:
                return torch.randn(shape, device=self.device)*self.sigma, False

    def certify(self, model, dataset, reuse = True, bs = 32, n0=128, n=100000, alpha=0.001, path=None, skip=20):
        """
        :param model: Base classifier mapping from [batch x channel x height x width] to [batch x num_classes]
        :param dataset: Certified dataset, one of cifar10, cifar100, imagenet
        :param reuse: Whether to reuse noise samples or not (faster for sound practice)
        :param bs: batch size 
        :param n0: number of samples used to estimate top1 class
        :param n: number of noise samples to evaluate smoothed classifier
        :param alpha: failure probability 
        :param path: path for imagenet validation set 
        :param skip: certify only one in #skip examples and skip the rest.
        """

        model.eval()
        if(isinstance(dataset, str)):
            if(path is None or dataset != 'imagenet' ):
                path = "./data"

            if(dataset == 'cifar10'):
                dataset = torchvision.datasets.CIFAR10(root=path, train=False, 
                        download=True, transform=torchvision.transforms.ToTensor())
            elif(dataset == 'cifar100'):
                dataset = torchvision.datasets.CIFAR100(root=path, train=False, 
                        download=True, transform=torchvision.transforms.ToTensor())
            elif(dataset == 'imagenet'):
                transform = torchvision.transforms.Compose([
                            torchvision.transforms.Resize(256),
                            torchvision.transforms.CenterCrop(224),
                            torchvision.transforms.ToTensor()
                        ])
                dataset = torchvision.datasets.ImageFolder(path, transform)
                
        X =  dataset[0][0]
        shape = [bs] + list(X.shape)
        counts = defaultdict(lambda: (Counter(), Counter()))
        true_labels = {}

        n0 = math.ceil(n0/bs)*bs 
        n = math.ceil(n/bs)*bs 

        with torch.no_grad():
            if(reuse == True):
                for bidx in tqdm(range((n+n0) // bs)):
                    noise, failure = self.get_noise_batch(shape)
                    if(failure and bidx >= n0//bs):
                        continue
                    for idx in range(0,len(dataset),skip):
                        X, y = dataset[idx]
                        preds = model((noise + X.to(self.device)).clip(-self.k, self.k+1)).argmax(-1)
                        counts[idx][bidx >= n0//bs].update(preds.detach().tolist())
                        true_labels[idx] = y
                        idx += 1
            else:
                for idx in tqdm(range(0,len(dataset),skip)):
                    X, y = dataset[idx]
                    true_labels[idx] = y
                    X = X.to(self.device)
                    for bidx in range((n+n0) // bs):
                        noise, failure = self.get_noise_batch(shape)
                        if(failure and bidx >= n0//bs):
                            continue
                        preds = model((noise + X).clip(-self.k, self.k+1)).argmax(-1)
                        counts[idx][bidx >= n0//bs].update(preds.detach().tolist())



        ret = [('true label', 'predicted label', 'r1', 'r2')]
        for idx in true_labels:
            pred = counts[idx][0].most_common(1)[0][0]
            top2 = counts[idx][1].most_common(2)
            ca = top2[0][1]
            if(len(top2) > 1):
                cb = top2[1][1]
            else:
                cb = 0

            r1 = (norm.ppf(proportion_confint(ca, n, alpha=alpha*2, method='beta')[0]) *
                  self.sigma * (top2[0][0] == pred == true_labels[idx]))

            r2 = ((norm.ppf(proportion_confint(ca, n, alpha=alpha, method='beta')[0]) -
                   norm.ppf(proportion_confint(cb, n, alpha=alpha, method='beta')[1])) *
                   self.sigma*(top2[0][0] == pred == true_labels[idx])/2)

            ret.append((true_labels[idx], pred, r1, r2))
        return ret 




    def set_thresholds(self):
        if(self.symbolic):
            try:
                import sympy            
                sigma = sympy.nsimplify(self.sigma, rational=True)
                x =  sympy.symbols('x')
                I = sympy.integrate(sympy.exp(-x*x/(2*sigma*sigma))/ sympy.sqrt(2*sympy.pi*sigma*sigma), x)*self.Q
                fun = lambda i: int((self.Q//2 + I.subs(x, sympy.Rational(2*i+1,510))).round())
            except:
                self.symbolic = False
                warnings.warn("import sympy failed, fall back to numerical evaluation of norm cdf")
        
        if(not self.symbolic):
            fun = lambda i:  math.ceil(norm.cdf((2*i+1)/510, scale=self.sigma)*self.Q)
        
        self.thresholds = [self.Q] * (2*(self.k+1)*255+1)   
        self.increments = [0]

        for idx, i in enumerate(range(-255*(self.k+1),  255*(self.k+1))):
            self.thresholds[idx] = fun(i)
            if(idx >= 1):
                self.increments.append(self.increments[-1] + int(self.thresholds[idx] != self.thresholds[idx-1]))
        self.increments.append(self.increments[-1])

        print(len(self.increments), len(self.thresholds))

