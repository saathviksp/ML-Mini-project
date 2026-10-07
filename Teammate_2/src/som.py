import numpy as np


class SelfOrganizingMap:
    """Kohonen Self-Organizing Map implemented from scratch in NumPy.

    A SOM is an unsupervised neural network that maps a high dimensional
    input space onto a low dimensional grid of nodes while preserving
    topology: inputs that are close together in the original space end up
    near each other on the grid.

    Every node on the (rows x cols) grid carries a weight vector, or
    codebook vector, of the same dimension as the input. Training is
    online and repeats three steps for each sample:

      1. Competition  - find the Best Matching Unit (BMU), the node whose
                        weight vector is nearest the input in Euclidean
                        distance.
      2. Cooperation  - decide how strongly each node is pulled, using a
                        Gaussian neighbourhood centred on the BMU and
                        measured in *grid* coordinates, not input space.
      3. Adaptation   - move the weights toward the input:
                        w <- w + lr(t) * h(t) * (x - w)

    Both the learning rate lr(t) and the neighbourhood width sigma(t)
    decay geometrically with time, so the map first unfolds globally and
    then fine-tunes locally.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions. A 5x5 grid gives 25 nodes.
    input_dim : int
        Number of input features (8 for HTRU2).
    learning_rate : float
        Initial learning rate lr0.
    sigma : float
        Initial neighbourhood radius sigma0. Defaults to half the larger
        grid side, so the first updates reach most of the map.
    n_epochs : int
        Number of passes over the training data.
    random_state : int
        Seed for weight initialisation and sample shuffling.
    """

    def __init__(self, rows=5, cols=5, input_dim=8, learning_rate=0.5,
                 sigma=None, n_epochs=100, random_state=42):
        self.rows = rows
        self.cols = cols
        self.input_dim = input_dim
        self.lr0 = learning_rate
        self.sigma0 = sigma if sigma is not None else max(rows, cols) / 2.0
        self.n_epochs = n_epochs
        self.random_state = random_state

        # Floors the decay schedules converge to by the final iteration.
        self.lr_min = 0.01
        self.sigma_min = 0.5

        self.weights = None
        self.quantization_errors_ = []

        # Grid coordinates of every node, shape (rows*cols, 2). Used for
        # the neighbourhood function, which works in grid space.
        grid_r, grid_c = np.meshgrid(np.arange(rows), np.arange(cols),
                                     indexing='ij')
        self.grid_coords = np.column_stack([grid_r.ravel(), grid_c.ravel()])

    @property
    def n_nodes(self):
        return self.rows * self.cols

    def _init_weights(self, X, rng):
        """Initialises codebook vectors by drawing random training rows.

        Sampling real observations places the map inside the data cloud
        from the start, which converges more reliably than uniform random
        initialisation.
        """
        idx = rng.choice(len(X), size=self.n_nodes, replace=True)
        self.weights = X[idx].astype(np.float64).copy()

    def _decay(self, start, end, t, total):
        """Geometric decay from `start` to `end` over `total` steps."""
        return start * (end / start) ** (t / total)

    def _find_bmu(self, x):
        """Returns the index of the node closest to x in Euclidean distance.

        Squared distance is enough here: it is monotone in distance, so
        the argmin is identical and the square root is skipped.
        """
        sq_dists = np.sum((self.weights - x) ** 2, axis=1)
        return int(np.argmin(sq_dists))

    def _neighbourhood(self, bmu_index, sigma):
        """Gaussian neighbourhood weights h(t) for every node.

        Distance is measured on the grid, so nodes that sit near the BMU
        on the map are pulled hardest regardless of where their weight
        vectors currently are in input space. This is what makes the SOM
        topology preserving.
        """
        grid_sq_dist = np.sum(
            (self.grid_coords - self.grid_coords[bmu_index]) ** 2, axis=1
        )
        return np.exp(-grid_sq_dist / (2.0 * sigma ** 2))

    def fit(self, X):
        """Trains the map with online (per-sample) updates."""
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        self._init_weights(X, rng)

        n_samples = len(X)
        total_steps = self.n_epochs * n_samples
        step = 0

        for _ in range(self.n_epochs):
            order = rng.permutation(n_samples)
            for i in order:
                x = X[i]
                lr = self._decay(self.lr0, self.lr_min, step, total_steps)
                sigma = self._decay(self.sigma0, self.sigma_min, step,
                                    total_steps)

                bmu = self._find_bmu(x)
                h = self._neighbourhood(bmu, sigma)

                # h[:, None] broadcasts the per-node scalar across all
                # input dimensions so every node moves toward x by its own
                # amount.
                self.weights += lr * h[:, None] * (x - self.weights)
                step += 1

            self.quantization_errors_.append(self.quantization_error(X))

        return self

    def predict(self, X):
        """Returns the BMU index for each sample, used as a cluster label."""
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._find_bmu(x) for x in X])

    def quantization_error(self, X):
        """Mean distance between each sample and its BMU weight vector.

        Falls as the map fits the data better, so plotting it per epoch is
        the standard convergence check for a SOM.
        """
        X = np.asarray(X, dtype=np.float64)
        labels = self.predict(X)
        diffs = X - self.weights[labels]
        return float(np.mean(np.linalg.norm(diffs, axis=1)))

    def u_matrix(self):
        """Mean distance from each node to its immediate grid neighbours.

        Plotted as a heatmap, high values trace ridges between regions of
        the map, which is how cluster boundaries are read off a SOM.
        """
        u = np.zeros((self.rows, self.cols))
        weights_grid = self.weights.reshape(self.rows, self.cols, -1)
        for r in range(self.rows):
            for c in range(self.cols):
                dists = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        dists.append(np.linalg.norm(
                            weights_grid[r, c] - weights_grid[nr, nc]
                        ))
                u[r, c] = float(np.mean(dists))
        return u

    def hit_map(self, X):
        """Counts how many samples map to each node, shaped like the grid."""
        labels = self.predict(X)
        counts = np.bincount(labels, minlength=self.n_nodes)
        return counts.reshape(self.rows, self.cols)
