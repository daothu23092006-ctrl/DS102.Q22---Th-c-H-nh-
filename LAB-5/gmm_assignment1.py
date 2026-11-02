import numpy as np
from gmm import GMM

real_means = [[2, 2], [8, 3], [3, 6]]
cov_matrix = [[1, 0], [0, 1]]

num_samples = 200
num_clusters = 3

data_1 = np.random.multivariate_normal(real_means[0], cov_matrix, num_samples)
data_2 = np.random.multivariate_normal(real_means[1], cov_matrix, num_samples)
data_3 = np.random.multivariate_normal(real_means[2], cov_matrix, num_samples)

dataset = np.vstack((data_1, data_2, data_3))

gmm = GMM(n_components=num_clusters)   # truyền n_components

gmm.fit(dataset)                       # không truyền K, không unpack tuple
print(f"Mean tìm được:\n{gmm.mean_vector}")
print(f"Mean thực tế:\n{real_means}")

# So sánh kết quả
print("\nNhận xét:")
print("GMM với EM tìm được các mean vector xấp xỉ với mean thực tế.")
print("Mỗi component tương ứng với một Gaussian trong dữ liệu.")
print(f"Mixing weights: {gmm.pi}")
