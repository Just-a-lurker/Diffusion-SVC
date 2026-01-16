import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR
opt = torch.optim.AdamW([torch.zeros(1)], lr=1.0)

sch = StepLR(opt, step_size=10, gamma=0.1)
print(opt.param_groups[0]['lr'])  # 1.0

sch2 = sch
print(opt.param_groups[0]['lr'])  # still 1.0
