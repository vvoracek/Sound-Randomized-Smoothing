# Sound-Randomized-Smoothing
Code for the paper Sound Randomized Smoothing in Floating-Point Arithmetics
https://arxiv.org/abs/2207.07209

The provided script contains normal distribution certification procedure for $\ell_2$ certified robustness discussed in the paper for sound randomized smoothing certification. 

the usage is as follows:

```python
from certifier import Certifier

Cert = Certifier(sigma=0.5)
certificates = Cert.certify(model, dataset='cifar10')
```

where ```model``` is a base classifier accepting a image tensors in the form NCHW

Additional parameters of ```Certifier``` and ```certify``` are documented in the script.
