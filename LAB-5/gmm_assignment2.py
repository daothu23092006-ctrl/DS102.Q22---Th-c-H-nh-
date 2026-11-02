import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from gmm import GMM

# Load ảnh
img = Image.open("cow.jpg").convert("RGB")
img_array = np.array(img)

height, width, channels = img_array.shape

pixels = img_array.reshape(-1, channels).astype(float)
pixels /= 255.0

# GMM phân cụm ảnh
model = GMM(n_components=2)
model.fit_fast(pixels) 

# Pixel thuộc cluster có xác suất cao nhất
pixel_cluster = np.argmax(model.r, axis=1)
pixel_cluster = pixel_cluster.reshape(height, width)

# Cluster có giá trị Green lớn nhất xem là background
centers = model.mean_vector
background_idx = np.argmax(centers[:, 1])

# Tạo mask foreground
foreground_mask = (pixel_cluster != background_idx)
binary_mask = foreground_mask.astype(np.uint8) * 255

# Loại bỏ background
filtered = img_array.copy()
filtered[~foreground_mask] = [0, 0, 0]

# Hiển thị
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(img_array)
ax[0].set_title("Original")

ax[1].imshow(binary_mask, cmap="gray")
ax[1].set_title("Foreground Mask")

ax[2].imshow(filtered)
ax[2].set_title("Background Removed")

plt.tight_layout()
plt.show()