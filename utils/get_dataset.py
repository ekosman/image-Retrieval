from torchvision import datasets, transforms
import torch


def get_dataset(args, normalize=True):
	if normalize:
		transform = transforms.Compose([
			transforms.Resize(size=32),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
	else:
		transform = transforms.Compose([
			transforms.Resize(size=32),
		])
	kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
	dataset = datasets.ImageFolder(args.image_path, transform=transform)
	loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
	return dataset, loader
