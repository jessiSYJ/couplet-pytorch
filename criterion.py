
import torch
import torch.nn as nn

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, predict, target, mask):
        # truncate to the same size
        target = target[:, :predict.size(1)]
        mask =  mask[:, :predict.size(1)]
        predict = to_contiguous(predict).view(-1, predict.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - predict.gather(1, target)
        output = output * mask
        output = torch.sum(output) / torch.sum(mask)

        return output