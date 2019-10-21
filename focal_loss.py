import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class FocalLoss(nn.Module):
	def __init__(self, gamma=0, alpha=None, size_average=True):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
		if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
		self.size_average = size_average

	def forward(self, x, target):
		if x.dim() > 2:
			x = x.view(x.size(0), x.size(1), -1)  # N,C,H,W => N,C,H*W
			x = x.transpose(1, 2)  # N,C,H*W => N,H*W,C
			x = x.contiguous().view(-1, x.size(2))  # N,H*W,C => N*H*W,C
		target = target.view(-1, 1)

		logpt = F.log_softmax(x)
		logpt = logpt.gather(1, target)
		logpt = logpt.view(-1)
		pt = Variable(logpt.data.exp())

		if self.alpha is not None:
			if self.alpha.type() != x.data.type():
				self.alpha = self.alpha.type_as(x.data)
			at = self.alpha.gather(0, target.data.view(-1))
			logpt = logpt * Variable(at)

		loss = -1 * (1 - pt) ** self.gamma * logpt
		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()
