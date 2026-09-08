"""Single-kernel permission probe; not evidence about the model bottleneck."""
import torch

x=torch.randn((1,2560),device='cuda',dtype=torch.bfloat16)
w=torch.randn((2560,19456),device='cuda',dtype=torch.bfloat16)
for _ in range(3): y=x@w
torch.cuda.synchronize()
torch.cuda.profiler.start()
y=x@w
torch.cuda.synchronize()
torch.cuda.profiler.stop()
print('Counter probe completed; this is not a model measurement.')
