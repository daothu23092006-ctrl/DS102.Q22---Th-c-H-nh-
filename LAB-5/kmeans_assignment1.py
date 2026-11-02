# Import thư viện
import numpy as np
from kmeans import KMeans
import matplotlib.pyplot as plt

# TẠO DỮ LIỆU GIẢ LẬP
true_centers = [[2, 2], [8, 3],[3, 6]]
cov_matrix = [[1, 0],[0, 1]]
samples_per_cluster = 200

cluster_A = np.random.multivariate_normal(true_centers[0], cov_matrix, samples_per_cluster)
cluster_B = np.random.multivariate_normal(true_centers[1], cov_matrix, samples_per_cluster)
cluster_C = np.random.multivariate_normal(true_centers[2], cov_matrix, samples_per_cluster)

# Ghép dữ liệu thành tập hoàn chỉnh
dataset = np.vstack([cluster_A, cluster_B, cluster_C])
num_clusters = 3

# HUẤN LUYỆN K-MEANS
kmeans_model = KMeans()
pred_labels, iterations = kmeans_model.fit(dataset, num_clusters)

# KẾT QUẢ
print("Số lần cập nhật centroid:", iterations)

print("\nCentroid sau khi hội tụ:")
print(kmeans_model.centroids[-1])

print("\nCentroid thực tế:")
print(np.array(true_centers))


'''
Đánh giá tác động của việc khởi tạo centroid ngẫu nhiên trong K-means

1. Tác động đến thời gian hội tụ
Centroid ban đầu được chọn ngẫu nhiên nên số vòng lặp có thể thay đổi.
Nếu vị trí khởi tạo gần tâm cụm thực tế, thuật toán thường hội tụ nhanh.
Ngược lại, centroid ban đầu không phù hợp sẽ làm tăng thời gian tìm nghiệm.

2. Tác động đến chất lượng phân cụm
K-means chỉ tìm nghiệm tối ưu cục bộ (local optimum),
không đảm bảo luôn tìm được nghiệm tối ưu toàn cục.

Một số trường hợp khởi tạo không tốt có thể gây:
- Nhiều centroid rơi vào cùng một cụm dữ liệu
- Một số cụm bị phân chia chưa chính xác
- Kết quả cuối cùng thay đổi giữa các lần chạy

Do đó random initialization ảnh hưởng trực tiếp
đến cả tốc độ hội tụ và độ chính xác phân cụm.
'''