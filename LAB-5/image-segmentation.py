import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from scr.GMM import GMM

gmm = GMM(n_components=2, max_iters=1000)

img = Image.open('image/cow.jpg').convert('RGB')
image = np.array(img)

H, W, C = image.shape
X = image.reshape(H * W, C)    

gmm.fit(X)

predict = gmm.predict(X)

mask = np.array(predict).reshape(H, W)
mask_numeric = (mask == "comp(0)").astype(int)

plt.figure(figsize=(8,8))
plt.imshow(mask_numeric, cmap='gray')  
plt.axis('off')
plt.show()