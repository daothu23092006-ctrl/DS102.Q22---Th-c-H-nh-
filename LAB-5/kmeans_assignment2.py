# Import thư viện
import numpy as np
from kmeans import KMeans
import matplotlib.pyplot as plt

cov = [[1, 0], [0, 1]]

X0 = np.random.multivariate_normal([2, 2], cov, 1200)
X1 = np.random.multivariate_normal([8, 3], cov, 200)
X2 = np.random.multivariate_normal([3, 6], cov, 1000)

X = np.concatenate((X0, X1, X2), axis=0)
K = 3

model = KMeans()
labels, it = model.fit(X, K)

print(f'Số vòng lặp: {it}')
print(f'Centroids tìm được:\n{model.centroids[-1]}')

final_labels = labels[-1]
print(f'Kích thước cluster dự đoán: {[int(np.sum(final_labels == i)) for i in range(K)]}')
print(f'Kích thước cluster thật: [1200, 200, 1000]')

'''
Đánh giá tác động của kích thước cluster không đồng đều đối với K-means

K-means vẫn hoạt động tốt nếu các cụm dữ liệu được phân tách rõ ràng trong không gian đặc trưng.
Khi số lượng điểm giữa các cluster chênh lệch, số vòng lặp để thuật toán hội tụ có thể biến động nhiều hơn so với trường hợp dữ liệu cân bằng.
Nếu dữ liệu vừa mất cân bằng về kích thước cluster vừa có hiện tượng chồng lấn (overlap), chất lượng phân cụm có thể giảm và dẫn đến kết quả phân loại chưa chính xác.
'''