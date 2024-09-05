import numpy as np
from sklearn.datasets import make_classification

class NoisyLabelSimulation:
    def __init__(self, n_samples=1000, n_features=15, noise_rate_positive=None, noise_rate_negative=None, noise_rate=None, correlation=0.8, seed=None):
        """
        Initialize the simulation class.
        
        - If `noise_rate` is provided, it will be used for both classes (for Importance Reweighting method).
        - If `noise_rate_positive` and `noise_rate_negative` are provided, these will be used for different classes (for Unbiased Estimator method).
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_rate_positive = noise_rate_positive
        self.noise_rate_negative = noise_rate_negative
        self.noise_rate = noise_rate  # Single noise rate for Importance Reweighting method
        self.correlation = correlation  # Correlation parameter
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # Consistent random generator
        self.X = None
        self.y = None
        self.y_noisy = None

    def generate_data(self):
        """
        Generate a synthetic dataset with noisy labels and correlated features.
        """
        # Step 1: Generate data with make_classification (10 informative, 3 redundant)
        X, y = make_classification(n_samples=self.n_samples, n_features=self.n_features, n_informative=10, 
                                   n_redundant=3, n_classes=2, flip_y=0, random_state=self.seed)
        y = np.where(y == 0, -1, 1)  # Remap labels to {-1, 1}
    
        # Step 2: Introduce correlation by making some features linear combinations of others
        X[:, 12] = self.correlation * X[:, 0] + (1 - self.correlation) * self.rng.normal(size=self.n_samples)
        X[:, 13] = self.correlation * X[:, 1] + (1 - self.correlation) * self.rng.normal(size=self.n_samples)
        X[:, 14] = self.correlation * X[:, 2] + (1 - self.correlation) * self.rng.normal(size=self.n_samples)

        # Step 3: Add noise to labels
        y_noisy = y.copy()
        indices_class_1 = np.where(y == 1)[0]
        indices_class_minus_1 = np.where(y == -1)[0]
    
        # Apply noise based on whether it's Unbiased Estimator, Label-Dependent Costs (with positive/negative rates) or Importance Reweighting (single noise rate)
        if self.noise_rate is not None:
            # Importance Reweighting: Single noise rate
            n_noisy = int(self.noise_rate * self.n_samples)
            
            # Ensure no class is fully eliminated
            if len(indices_class_1) > 1:
                noise_class_1 = np.random.choice(indices_class_1, size=min(len(indices_class_1) - 1, int(n_noisy / 2)), replace=False)
                y_noisy[noise_class_1] = -1
            
            if len(indices_class_minus_1) > 1:
                noise_class_minus_1 = np.random.choice(indices_class_minus_1, size=min(len(indices_class_minus_1) - 1, int(n_noisy / 2)), replace=False)
                y_noisy[noise_class_minus_1] = 1

        elif self.noise_rate_positive is not None and self.noise_rate_negative is not None:
            # Unbiased Estimator: Separate noise rates for positive and negative classes
            n_flip_class_1 = int(self.noise_rate_positive * len(indices_class_1))
            n_flip_class_minus_1 = int(self.noise_rate_negative * len(indices_class_minus_1))

            if len(indices_class_1) > 1:
                flip_indices_class_1 = self.rng.choice(indices_class_1, size=n_flip_class_1, replace=False)
                y_noisy[flip_indices_class_1] = -y_noisy[flip_indices_class_1]

            if len(indices_class_minus_1) > 1:
                flip_indices_class_minus_1 = self.rng.choice(indices_class_minus_1, size=n_flip_class_minus_1, replace=False)
                y_noisy[flip_indices_class_minus_1] = -y_noisy[flip_indices_class_minus_1]
                        
        # Store the generated data
        self.X, self.y, self.y_noisy = X, y, y_noisy

    def get_data(self):
        """
        Return the generated data.
        """
        self.generate_data()
        return self.X, self.y, self.y_noisy

    def generate_multiple_datasets(self, n_datasets=100):
        """
        Generate multiple datasets for simulation.
        """
        datasets = []
        for i in range(n_datasets):
            self.rng = np.random.default_rng(self.seed + i)  # Update the seed for each dataset, while keeping it consistent
            datasets.append(self.get_data())
        return datasets