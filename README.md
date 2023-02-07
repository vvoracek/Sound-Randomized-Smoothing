# Sound-Randomized-Smoothing
Code for the paper Sound Randomized Smoothing in Floating-Point Arithmetics
https://arxiv.org/abs/2207.07209

The provided script contain normal distribution samplers discussed in the paper for sound randomized smoothing certification. It can be used as a drop-in replacement to the standard (unsound) practices.

the usage is as follows:

```python
from certifier import Certifier
certifier = Certifier(sigma=0.5)
certificates = certifier.certify(model, dataset='cifar10')
```

where ```python model ``` is a base classifier accepting a image tensors in the form NCHW