import numpy as np

def calculate_entropy(y):
    """Calculates Shannon entropy of label vector y."""
    n = len(y)
    if n == 0:
        return 0.0
    p1 = np.sum(y == 1) / n
    p0 = 1.0 - p1
    if p0 == 0 or p1 == 0:
        return 0.0
    return - (p0 * np.log2(p0) + p1 * np.log2(p1))

class Node:
    """Decision Tree Node."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, proba=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.proba = proba

    def is_leaf(self):
        return self.value is not None

class DecisionTreeScratch:
    """Decision Tree built from scratch using Information Gain."""
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        n_features = X.shape[1]
        if self.max_features is None:
            self.n_features_split = n_features
        elif isinstance(self.max_features, str) and self.max_features == 'sqrt':
            self.n_features_split = int(np.sqrt(n_features))
        elif isinstance(self.max_features, int):
            self.n_features_split = min(self.max_features, n_features)
        else:
            self.n_features_split = int(np.sqrt(n_features))

        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_proba = np.mean(y) if n_samples > 0 else 0.0
            leaf_val = 1 if leaf_proba >= 0.5 else 0
            return Node(value=leaf_val, proba=leaf_proba)

        # Select random feature subset
        feat_idxs = np.random.choice(n_features, self.n_features_split, replace=False)
        best_feat, best_thresh, best_ig = self._best_split(X, y, feat_idxs)

        if best_ig <= 1e-7 or best_feat is None:
            leaf_proba = np.mean(y)
            leaf_val = 1 if leaf_proba >= 0.5 else 0
            return Node(value=leaf_val, proba=leaf_proba)

        # Split data
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _best_split(self, X, y, feat_idxs):
        best_ig = -1.0
        split_idx, split_thresh = None, None
        parent_entropy = calculate_entropy(y)
        n = len(y)

        for feat in feat_idxs:
            X_column = X[:, feat]
            thresholds = np.percentile(X_column, np.linspace(10, 90, 15))
            for thresh in thresholds:
                left_mask = X_column <= thresh
                n_left = np.sum(left_mask)
                n_right = n - n_left

                if n_left == 0 or n_right == 0:
                    continue

                e_left = calculate_entropy(y[left_mask])
                e_right = calculate_entropy(y[~left_mask])
                child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

                ig = parent_entropy - child_entropy
                if ig > best_ig:
                    best_ig = ig
                    split_idx = feat
                    split_thresh = thresh

        return split_idx, split_thresh, best_ig

    def _predict_sample_proba(self, node, x):
        if node.is_leaf():
            return node.proba
        if x[node.feature] <= node.threshold:
            return self._predict_sample_proba(node.left, x)
        return self._predict_sample_proba(node.right, x)

    def predict_proba(self, X):
        return np.array([self._predict_sample_proba(self.root, x) for x in X])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class RandomForestScratch:
    """Random Forest Classifier implemented from scratch in Python/NumPy."""
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features='sqrt', random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        n_samples = X.shape[0]

        for i in range(self.n_estimators):
            # Bootstrap sampling with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]

            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        """Ensemble average predicted probability across trees."""
        tree_probas = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(tree_probas, axis=0)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
