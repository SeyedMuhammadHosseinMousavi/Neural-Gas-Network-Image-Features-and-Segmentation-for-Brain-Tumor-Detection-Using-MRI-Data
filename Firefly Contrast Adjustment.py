import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, exposure

class FireflyAlgorithm:
    def __init__(self, n_fireflies=10, max_iter=50, alpha=0.2, beta=1, gamma=1):
        self.n_fireflies = n_fireflies
        self.max_iter = max_iter
        self.alpha = alpha  # Mutation Coefficient
        self.beta = beta  # Attraction Coefficient Base Value
        self.gamma = gamma  # Light Absorption Coefficient

    def optimize(self, initial_params, objective_function):
        # Initialize fireflies randomly
        fireflies = np.random.uniform(low=0.5, high=2.0, size=(self.n_fireflies, len(initial_params)))
        intensities = np.array([objective_function(f) for f in fireflies])

        for t in range(self.max_iter):
            print(f"Iteration {t + 1}/{self.max_iter}: Best Cost = {intensities.min()}")
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if intensities[j] < intensities[i]:
                        distance = np.linalg.norm(fireflies[i] - fireflies[j])
                        beta_attraction = self.beta * np.exp(-self.gamma * (distance ** 2))
                        mutation = self.alpha * (np.random.uniform(-0.1, 0.1, len(initial_params)))

                        fireflies[i] += beta_attraction * (fireflies[j] - fireflies[i]) + mutation
                        fireflies[i] = np.clip(fireflies[i], 0.5, 2.0)  # Clip to reasonable parameter range

                        intensities[i] = objective_function(fireflies[i])

            self.alpha *= 0.98  # Damping mutation coefficient

        best_firefly_idx = np.argmin(intensities)
        return fireflies[best_firefly_idx]

# Objective function for contrast adjustment
def objective_function(params):
    gamma, gain = params
    adjusted = exposure.adjust_gamma(test_image, gamma=gamma, gain=gain)
    return -np.var(adjusted)  # Maximize variance for better contrast

# Main function to apply Firefly Algorithm for image contrast enhancement
def enhance_contrast_with_firefly(image_path, n_fireflies=10, max_iter=100, alpha=0.2, beta=1, gamma=1):
    # Load the image
    image = img_as_float(io.imread(image_path))
    if len(image.shape) == 3:  # Convert to grayscale if it's a color image
        image = np.mean(image, axis=2)

    global test_image  # Set the test image globally for the objective function
    test_image = image

    # Initialize Firefly Algorithm
    fa = FireflyAlgorithm(n_fireflies=n_fireflies, max_iter=max_iter, alpha=alpha, beta=beta, gamma=gamma)

    # Optimize parameters for gamma adjustment
    initial_params = [1.0, 1.0]  # Initial gamma and gain
    best_params = fa.optimize(initial_params, objective_function)
    optimized_gamma, optimized_gain = best_params

    # Apply optimized gamma correction
    enhanced_image = exposure.adjust_gamma(image, gamma=optimized_gamma, gain=optimized_gain)

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title(f'Enhanced Image (Gamma={optimized_gamma:.2f}, Gain={optimized_gain:.2f})')
    plt.axis('off')

    plt.show()

    return enhanced_image

# Example usage
if __name__ == "__main__":
    image_path = 'tst2.jpg'  # Replace with your image path
    enhanced_image = enhance_contrast_with_firefly(image_path)