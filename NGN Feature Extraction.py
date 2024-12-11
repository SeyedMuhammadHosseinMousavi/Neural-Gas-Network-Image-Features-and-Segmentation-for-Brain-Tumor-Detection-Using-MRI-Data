import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from sklearn.preprocessing import StandardScaler

class NeuralGas:
    def __init__(self, n_units=10, max_iter=100, eta_start=0.5, eta_end=0.01, lambda_start=30, lambda_end=0.1):
        self.n_units = n_units
        self.max_iter = max_iter
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.units = None

    def _update_learning_rate(self, i):
        return self.eta_start * (self.eta_end / self.eta_start) ** (i / self.max_iter)

    def _update_neighborhood_range(self, i):
        return self.lambda_start * (self.lambda_end / self.lambda_start) ** (i / self.max_iter)

    def train(self, data):
        n_samples, n_features = data.shape
        self.units = np.random.rand(self.n_units, n_features)

        for i in range(self.max_iter):
            # Print iteration
            print(f"Iteration {i + 1}/{self.max_iter}")
            
            eta = self._update_learning_rate(i)
            lambd = self._update_neighborhood_range(i)

            indices = np.random.permutation(n_samples)
            for index in indices:
                x = data[index]
                dists = np.linalg.norm(self.units - x, axis=1)
                ranking = np.argsort(dists)
                
                for rank, idx in enumerate(ranking):
                    influence = np.exp(-rank / lambd)
                    self.units[idx] += eta * influence * (x - self.units[idx])

    def extract_features(self, data):
        dists = np.linalg.norm(self.units[:, np.newaxis, :] - data, axis=2)
        features = np.min(dists, axis=0)
        return features

# Load an image
image_path = 'tst2.jpg'  # Replace with your image path
image = img_as_float(io.imread(image_path))
if len(image.shape) == 3:  # Convert to grayscale if it's a color image
    image = np.mean(image, axis=2)

pixels = image.reshape(-1, 1)  # Reshape image to a 2D array of pixels

# Normalize pixel data
scaler = StandardScaler()
pixels_normalized = scaler.fit_transform(pixels)

# Train Neural Gas
ng = NeuralGas(n_units=10, max_iter=10)
ng.train(pixels_normalized)

# Extract features
features = ng.extract_features(pixels_normalized)
features_image = features.reshape(image.shape)

# Plot the original image and the extracted features
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(features_image, cmap='viridis')
plt.title('Extracted Features (NGN)')
plt.axis('off')

plt.show()