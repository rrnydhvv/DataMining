# ===========================================================================
# THUẬT TOÁN K-MEANS CLUSTERING (PHÂN CỤM)
# ===========================================================================

import csv
import math
from typing import List, Dict, Tuple, Any

# ---------------------------------------------------------------------------
# ĐỌC DỮ LIỆU
# ---------------------------------------------------------------------------
def read_data(file_path: str) -> Tuple[List[str], List[List[str]]]:
    """Đọc file CSV và trả về header và dữ liệu."""
    data: List[List[str]] = []
    header: List[str] = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) > 0:
                clean_row = [value.strip().lower() for value in row]
                data.append(clean_row)
    return header, data


# ---------------------------------------------------------------------------
# MÃ HÓA ONE-HOT ENCODING (OHE)
# ---------------------------------------------------------------------------
def create_ohe_encoding_table(data: List[List[str]], cols: List[int]) -> Tuple[Dict[int, Dict[str, List[int]]], List[str]]:
    """Tạo bảng mã hóa One-Hot Encoding cho các cột categorical."""
    encoding_table: Dict[int, Dict[str, List[int]]] = {}
    ohe_col_names: List[str] = []

    for col in cols:
        #tìm các giá trị duy nhất trong cột
        unique_values: List[str] = []
        for row in data:
            if row[col] not in unique_values:
                unique_values.append(row[col])
        unique_values.sort() #sắp xếp các giá trị duy nhất để đảm bảo thứ tự mã hóa ổn định

        num_values = len(unique_values) #đếm số giá trị
        encoding_table[col] = {}
        
        for i, value in enumerate(unique_values):
            vector = [0] * num_values
            vector[i] = 1
            encoding_table[col][value] = vector

        for value in unique_values:
            ohe_col_names.append(f"{value}")

    return encoding_table, ohe_col_names


def encode_data_ohe(data: List[List[str]], encoding_table: Dict[int, Dict[str, List[int]]], cols: List[int]) -> List[List[float]]:
    """Chuyển dữ liệu categorical thành số bằng One-Hot Encoding."""
    numeric_data: List[List[float]] = []
    for row in data:
        vector: List[float] = []
        for col in cols:
            value = row[col]
            vector.extend(float(v) for v in encoding_table[col][value]) #tra cứu vector mã hóa và thêm vào vector dữ liệu
        numeric_data.append(vector)
    return numeric_data


# ---------------------------------------------------------------------------
# KHOẢNG CÁCH EUCLID
# ---------------------------------------------------------------------------
def euclidean_distance(point_a: List[float], point_b: List[float]) -> float:
    """Tính khoảng cách Euclid giữa 2 điểm."""
    total: float = 0.0
    for i in range(len(point_a)):
        total += (point_a[i] - point_b[i]) ** 2
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# K-MEANS CORE FUNCTIONS
# ---------------------------------------------------------------------------
def initialize_centroids(numeric_data: List[List[float]], k: int) -> List[List[float]]:
    """Chọn K điểm đầu tiên làm tâm cụm ban đầu."""
    centroids: List[List[float]] = []
    for i in range(k):
        centroids.append(numeric_data[i][:])
    return centroids


def assign_clusters(numeric_data: List[List[float]], centroids: List[List[float]]) -> List[int]:
    """Gán mỗi điểm dữ liệu vào cụm có tâm cụm gần nhất."""
    cluster_assignments: List[int] = [] #chứa chỉ số cụm của mỗi điểm dữ liệu
    for point in numeric_data: #duyệt qua từng điểm dữ liệu
        min_dist: float = float('inf')
        closest_cluster: int = 0

        for j, centroid in enumerate(centroids):
            dist = euclidean_distance(point, centroid)
            if dist < min_dist:
                min_dist = dist
                closest_cluster = j

        cluster_assignments.append(closest_cluster)
    return cluster_assignments


def update_centroids(numeric_data: List[List[float]], cluster_assignments: List[int], k: int, num_dims: int) -> List[List[float]]:
    """Tính tâm cụm mới = trung bình các điểm trong mỗi cụm."""
    new_centroids: List[List[float]] = []
    for j in range(k): #duyệt qua từng cụm
        #danh sách các điểm thuộc cụm j
        cluster_points = [numeric_data[i] for i in range(len(numeric_data)) if cluster_assignments[i] == j]

        #khi 1 cụm không có điểm nào, đặt tâm cụm mới là vector 0 để tránh lỗi chia cho 0
        if len(cluster_points) == 0:
            new_centroids.append([0.0] * num_dims)
        else:
            centroid: List[float] = []
            for d in range(num_dims):
                mean_val = sum(point[d] for point in cluster_points) / len(cluster_points)
                centroid.append(round(mean_val, 4))
            new_centroids.append(centroid)

    return new_centroids


def has_converged(old_centroids: List[List[float]], new_centroids: List[List[float]], threshold: float = 0.0001) -> bool:
    """Kiểm tra xem thuật toán đã hội tụ chưa."""
    total_shift: float = 0.0 #tọa độ dịch chuyển tổng của tất cả các tâm cụm
    for i in range(len(old_centroids)):
        #tính khoảng cách dịch chuyển của tâm cụm i từ old_centroids[i] đến new_centroids[i]
        total_shift += euclidean_distance(old_centroids[i], new_centroids[i])
    return total_shift < threshold


def kmeans_algorithm(numeric_data: List[List[float]], k: int, max_iterations: int = 20) -> Tuple[List[int], List[List[float]], List[Dict[str, Any]]]:
    """Chạy thuật toán K-Means."""
    #lây số chiều của dữ liệu
    num_dims = len(numeric_data[0])
    history: List[Dict[str, Any]] = []

    centroids = initialize_centroids(numeric_data, k)

    for iteration in range(1, max_iterations + 1):
        cluster_assignments = assign_clusters(numeric_data, centroids)
        new_centroids = update_centroids(numeric_data, cluster_assignments, k, num_dims)

        history.append({
            'vong': iteration,
            'tam_cum': [t[:] for t in centroids],
            'tam_moi': [t[:] for t in new_centroids],
            'phan_cum': cluster_assignments[:]
        })

        if has_converged(centroids, new_centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    return cluster_assignments, centroids, history


# ---------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------
def main() -> None:
    print("--- THUAT TOAN K-MEANS CLUSTERING ---")
    file_path = "golf_df.csv"
    header, data = read_data(file_path)
    label_col = len(header) - 1

    print(f"[Du lieu] Tong so mau: {len(data)}")

    cols = [i for i in range(len(header)) if i != label_col]
    encoding_table, ohe_col_names = create_ohe_encoding_table(data, cols)
    
    numeric_data = encode_data_ohe(data, encoding_table, cols)
    num_dims = len(numeric_data[0])

    print(f"\n[Buoc 0] Ma hoa du lieu (OHE): {len(cols)} thuoc tinh => {num_dims} chieu.")

    num_clusters = 2
    print(f"\n[K-Means] So cum K = {num_clusters}")

    cluster_assignments, final_centroids, history = kmeans_algorithm(numeric_data, num_clusters)

    print(f"\n[Qua trinh phan cum]")
    for log_entry in history:
        iteration = log_entry['vong']
        print(f"  Vong {iteration}:")
        
        for j in range(num_clusters):
            pts = [i + 1 for i in range(len(numeric_data)) if log_entry['phan_cum'][i] == j]
            print(f"    Cum {j+1}: {len(pts)} diem -> Tam moi: {log_entry['tam_moi'][j]}")

        if iteration == history[-1]['vong']:
            print("  => DA HOI TU!")

    print(f"\n[Ket qua cuoi cung]")
    for j in range(num_clusters):
        print(f"  Cum {j+1}:")
        for i in range(len(data)):
            if cluster_assignments[i] == j:
                original_row = data[i]
                print(f"    Mau {i+1}: {', '.join(original_row[:label_col])} | Play={original_row[label_col]}")

    print("\n[So sanh voi nhan thuc te (Tham khao)]")
    for j in range(num_clusters):
        label_counts: Dict[str, int] = {}
        for i in range(len(data)):
            if cluster_assignments[i] == j:
                label = data[i][label_col]
                label_counts[label] = label_counts.get(label, 0) + 1

        total = sum(label_counts.values())
        print(f"  Cum {j+1} ({total} mau):")
        for label, count in label_counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"    Play={label}: {count} mau ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
