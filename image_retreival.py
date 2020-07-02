import argparse
import torch
from knn.knn import KNN
from utils.get_cifar100 import get_cifar100
from utils.get_dataset import get_dataset
from utils.logger import register_logger
from utils.show_neighbors import show_neighbors
from vgg_utils import VGG
from tqdm import tqdm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Image retrieval on CIFAR-100 using VGG-16 embedding and KNN')
parser.add_argument('-vgg_path', type=str, default=r'checkpoint/vgg16/vgg16-190-best.pth',
                    help='Path to vgg-16 pretrained model on cifar-100')
parser.add_argument('-image_path', type=str, default=r'test_images',
                    help='Path to directory containing images for which we want to retrieve images from cifar-100')
parser.add_argument('-batch_size', type=int, default=100,
                    help='Path to directory containing images for which we want to retrieve images from cifar-100')


def get_features(model, loader):
    features = []
    i=0
    with torch.no_grad():
        for batch, _ in tqdm(loader):
            if torch.cuda.is_available():
                batch = batch.cuda()
            b_features = model(batch).detach().cpu().numpy()
            for f in b_features:
                i+= 1
                features.append(f)
            if i > 20: break

    return features


if __name__ == '__main__':
    args = parser.parse_args()
    register_logger()

    ret_dataset, ret_loader = get_dataset(args)
    ret_paths = [a[0] for a in ret_dataset.imgs]
    cifar_dataset, cifar_loader = get_cifar100(args)
    cifar_images = cifar_dataset.data

    vgg = VGG(use_cuda=torch.cuda.is_available(), pretrained_path=args.vgg_path)

    cifar_features = get_features(model=vgg, loader=cifar_loader)
    ret_features = get_features(model=vgg, loader=ret_loader)

    knn = KNN()
    knn.fit(x=cifar_images, x_features=cifar_features)
    rets_neighbors = knn.predict(ret_features, k=7)

    for i, (orig_path, neighbors) in enumerate(zip(ret_paths, rets_neighbors)):
        neighbors_imgs = [knn[n] for n in neighbors]
        show_neighbors(orig=plt.imread(orig_path), neighbors=neighbors_imgs, name=str(i))


