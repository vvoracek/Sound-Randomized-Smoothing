# Sound-Randomized-Smoothing
Code for the paper Sound Randomized Smoothing in Floating-Point Arithmetics
https://arxiv.org/abs/2207.07209

The provided script contain samplers discussed in the paper for sound randomized smoothing certification. It can be used as a drop-in replacement to the standard (unsound) practices.

Instead of:

```python
inp = x + torch.randn(x.shape, device='cuda')*self.sigma
```

use:

```python
from samplers import Sampler
sampler = Sampler(self.sigma, shape_of_batch, n)
...
inp = sampler.sample_noise(x)
```

where  $n$ is the number of samples used to estimate the output of the smoothed classifier.
