import matplotlib.pyplot as plt
from os import path, makedirs


def show_neighbors(orig, neighbors, name):
	if not path.exists('output'):
		makedirs('output', exist_ok=True)

	f, axarr = plt.subplots(4, 2)
	image_datas = [orig] + neighbors
	for i, ax in enumerate(axarr.flatten()):
		if i == 0:
			ax.set_title("Original image")
		else:
			ax.set_title(f"Neighbor {i}")
		ax.imshow(image_datas[i])

	plt.savefig(path.join('output', f"{name}.png"))

