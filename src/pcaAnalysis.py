import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load images into a list
image_files = [f'{i}.png' for i in range(1000)]
images = [plt.imread("data/mujoco/rgb/"+img) for img in image_files]

def flatten_images(images):
    image_data = []
    for idx, img in enumerate(images):
        try:
            image_data.append(img.flatten())
        except Exception as e:
            print(f"Failed to flatten image at index {idx}: {e}")
    return np.array(image_data)

# Flatten images and create the data matrix
image_data = np.array([img.flatten() for img in [images[0]]])

# Standardize the data
scaler = StandardScaler()
standardized_data = scaler.fit_transform(image_data)

# Perform PCA
number_of_components = 50  # Adjust as needed
pca = PCA(n_components=number_of_components)
principal_components = pca.fit_transform(standardized_data)

# Optional: Visualize the first principal component as an image
first_pc_image = pca.components_[0].reshape(images[0].shape)
plt.imshow(first_pc_image, cmap='gray')
plt.title("First Principal Component")

# Save the figure
# plt.savefig("first_principal_component.png")

# Plot explained variance ratio
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.grid()
plt.savefig("variance_by_pcs_rbg.png")

# Optionally, close the plot to free memory
plt.close()