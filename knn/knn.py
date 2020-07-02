import numpy as np


class KNN:
	def __init__(self):
		self.images = []
		self.features = []

	def __len__(self):
		return len(self.images)

	def fit(self, x, x_features):
		"""
		:param x: (n_samples, sample_dim)
		"""
		self.images = x
		self.features = x_features

	def __getitem__(self, item):
		return self.images[item]

	def predict(self, x, k):
		if len(self) == 0:
			return None

		dists = np.array([self.single_image_distances(x_, k) for x_ in x])
		sorted_idx = np.argsort(a=dists, axis=1)
		return sorted_idx[:, :k]

	def single_image_distances(self, x, k):
		if k > len(self):
			return [list(range(len(self)))]
		return [np.linalg.norm(x - f) for f in self.features]
