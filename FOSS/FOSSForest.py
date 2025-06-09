import math
import numpy as np
import FOSSTree

class FOSSForest:
    def __init__(self, n_trees, subset_size, Smin=1):
        self.n_trees = n_trees
        self.subset_size = subset_size
        self.Smin = Smin
        self.trees = []

    def fit(self, X):
        n_samples = X.shape[0]
        hmax = math.ceil(math.log2(self.subset_size))
        self.trees = []

        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, self.subset_size, replace=True)
            subset = X[indices]
            tree = FOSSTree(subset, h=0, hmax=hmax, Smin=self.Smin)
            self.trees.append(tree)

    def get_trees(self):
        return self.trees
