import torch

def noise_input(t, dim, mean=0, std=1):
  """
  returns a noisy white noise
  `t` ignored and is just for integrity
  """
  return torch.normal(mean=mean, std=std, size=(dim,))

def step_input(t, dim, interval0=10, interval1=10, amp=20, noise=False, noise_mean=0, noise_std=1):
  """
  returns a periodic step input
  Parameters:
  -----
  `interval0`: int
  the interval of 0
  `interval1`: int
  the interval of 1
  `amp`: number
  the amplitude of spikes

  Returns:
  -----
  a spike input for `dim` neurons in the `t`'s second
  """
  if (t % (interval0 + interval1)) > interval0:
    ans = torch.ones(dim) * amp
  else:
    ans = torch.zeros(dim)
  if noise:
    ans += noise_input(t, dim, noise_mean, noise_std)
  return ans

def sin_input(t, dim, step=1/6, amp=20, noise=False, noise_mean=0, noise_std=1):
  """
  returns a periodic step input
  Parameters:
  -----
  `step`: float
  the steps of sin function
  `amp`: number
  the amplitude of spikes

  Returns:
  -----
  a spike input for `dim` neurons in the `t`'s second
  """
  ans = (torch.sin(torch.ones(dim) * torch.pi * t * step) + 1) * amp
  if noise:
    ans += noise_input(t, dim, noise_mean, noise_std)
  return ans

class RandomPattern:
    def __init__(self, size=None, pattern1=None, pattern2=None, period=1):
        if pattern1 == None and pattern2 == None:
            if size == None:
                raise ValueError("Whether patterns or the size must be given")
            pattern1 = (torch.rand(size) > 0.5).type(torch.int8)
            pattern2 = (torch.rand(size) > 0.5).type(torch.int8)
        else:
            pattern1 = torch.tensor(pattern1)
            pattern2 = torch.tensor(pattern2)
            
        self.pats = [pattern1, pattern2]
        self.period = period
            
        

    def __len__(self) -> int:
        return len(self.pats)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        patInd = (index // self.period) % 2
        if index == 0:
            print()
        return self.pats[patInd].squeeze(), patInd
    def __call__(self, t, dim) -> torch.Tensor:
       return self.__getitem__(t)[0]
