import torch

import torchvision
from torchvision import transforms


def get_cifar100(args):
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	dataset = torchvision.datasets.CIFAR100(
		root=r'./data', train=True, download=True, transform=transform)
	kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
	loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
	return dataset, loader
