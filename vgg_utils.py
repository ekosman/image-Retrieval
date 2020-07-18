from torch import nn, load
from torchvision import transforms, models
from torchvision.models.vgg import model_urls

class VGG:
	def __init__(self, use_cuda=False, pretrained_path=None):
		self.use_cuda = use_cuda
		model_urls['vgg19'] = model_urls['vgg19'].replace('https://', 'http://')
		model = models.vgg19(pretrained=True, progress=True)

		model.classifier = nn.Sequential(*list(model.classifier.children())[:3])
		self.model = model
		if use_cuda:
			self.model = self.model.cuda()

		self.model = self.model.eval()

	def __call__(self, x):
		return self.model(x)
