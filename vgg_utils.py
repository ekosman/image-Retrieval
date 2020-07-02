from torch import nn, load
from torchvision import transforms, models


class VGG:
	def __init__(self, use_cuda=False, pretrained_path=None):
		self.use_cuda = use_cuda
		model = models.vgg16_bn(pretrained=True)
		# num_ftrs = model.classifier[6].in_features
		# model.classifier[6] = nn.Linear(num_ftrs, 100)
		# if pretrained_path is not None:
		# 	model.load_state_dict(load('checkpoint/vgg16/vgg16-190-best.pth', map_location='cpu'))
		#
		# model.classifier = nn.Sequential(*list(model.classifier.children())[:3])
		self.model = nn.Sequential(*list(model.features.children())[:2])
		if use_cuda:
			self.model = self.model.cuda()

		self.model = self.model.eval()

	def __call__(self, x):
		return self.model(x)
