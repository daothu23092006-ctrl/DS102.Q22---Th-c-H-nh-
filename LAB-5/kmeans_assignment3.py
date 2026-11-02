import numpy as np
from kmeans import KMeans
import matplotlib.pyplot as plt

cov1 = [[1, 0], [0, 1]]
cov2 = [[10, 0], [0, 1]]   
N = 200

# Mỗi dòng sinh ra array shape (200, 2)
X0 = np.random.multivariate_normal([2, 2], cov1, N)
X1 = np.random.multivariate_normal([8, 3], cov1, N)
X2 = np.random.multivariate_normal([3, 6], cov2, N)  

X = np.concatenate((X0, X1, X2), axis=0)

K = 3
true_centers = [[2, 2], [8, 3], [3, 6]]

for seed in [0, 1, 5, 42, 99]:
    np.random.seed(seed)
    model = KMeans()
    labels, it = model.fit(X, K)

    print(f"Seed {seed}: {it} vòng lặp")
    print(f"Centroids:\n{model.centroids[-1]}\n")

# In kết quả cuối
print(f"Số vòng lặp: {it}")
print(f"Centroids tìm được:\n{model.centroids[-1]}")
print(f"Centroids thật:\n{true_centers}")


'''
Ảnh hưởng của covariance matrix lớn

Centroid tìm được có thể lệch nhiều hơn so với tâm thực tế.
Số vòng lặp hội tụ biến động giữa các lần chạy do cluster trải rộng làm việc phân cụm khó hơn.
Covariance lớn tạo cụm dạng elipse, gây overlap và làm ranh giới giữa các cluster kém rõ ràng.
K-means chỉ dựa trên khoảng cách Euclidean, nên thường không xử lý tốt dữ liệu có hình dạng cụm phức tạp.
'''