# ===========================================================================
# THUẬT TOÁN K-MEANS CLUSTERING (PHÂN CỤM)
# ===========================================================================
# Mô tả:
#   K-Means là thuật toán PHÂN CỤM (clustering) - HỌC KHÔNG GIÁM SÁT
#   (unsupervised learning), KHÔNG phải phân loại (classification).
#   Thuật toán chia dữ liệu thành K cụm dựa trên khoảng cách.
#
# LƯU Ý QUAN TRỌNG:
#   - K-Means làm việc với DỮ LIỆU SỐ (numerical data).
#   - Dataset golf_play là dữ liệu PHÂN LOẠI (categorical).
#   - Để áp dụng K-Means, ta cần MÃ HÓA dữ liệu categorical => số (encoding).
#   - Đây là ví dụ minh họa, trong thực tế K-Means không phải lựa chọn
#     tốt nhất cho dữ liệu hoàn toàn categorical.
#
# Công thức khoảng cách Euclid:
#   d(x, y) = sqrt( sum( (xi - yi)^2 ) )
#
# Quy trình K-Means:
#   Bước 1: Chọn K tâm cụm ban đầu (centroids)
#   Bước 2: Gán mỗi điểm dữ liệu vào cụm có tâm gần nhất
#   Bước 3: Cập nhật tâm cụm = trung bình các điểm trong cụm
#   Bước 4: Lặp lại Bước 2-3 cho đến khi hội tụ
#            (các cụm không thay đổi hoặc đạt số vòng lặp tối đa)
# ===========================================================================

import csv
import math

# ---------------------------------------------------------------------------
# BUOC 0: Doc du lieu tu file CSV
# ---------------------------------------------------------------------------
def read_data(file_path):
    """
    Doc file CSV va tra ve tieu_de (header) va du_lieu (list cac dong).
    """
    data = []
    header = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) > 0:
                clean_row = [value.strip().lower() for value in row]
                data.append(clean_row)
    return header, data


# ---------------------------------------------------------------------------
# MA HOA DU LIEU CATEGORICAL => SO (One-Hot Encoding - OHE)
# ---------------------------------------------------------------------------
# Vi du lieu golf_play la categorical, ta can chuyen doi sang so.
#
# ONE-HOT ENCODING (Ma hoa nhi phan):
#   Moi gia tri cua thuoc tinh duoc bieu dien bang 1 vector nhi phan
#   co dung 1 phan tu = 1, cac phan tu con lai = 0.
#
#   Vi du thuoc tinh Outlook co 3 gia tri {overcast, rainy, sunny}:
#     overcast => [1, 0, 0]
#     rainy    => [0, 1, 0]
#     sunny    => [0, 0, 1]
#
#   Vi du thuoc tinh Windy co 2 gia tri {false, true}:
#     false    => [1, 0]
#     true     => [0, 1]
#
# UU DIEM cua OHE so voi Label Encoding:
#   - Khong tao ra quan he thu tu gia giua cac gia tri
#     (Label Encoding: sunny=2 > rainy=1 > overcast=0 => sai y nghia)
#   - Khoang cach Euclid giua bat ky 2 gia tri khac nhau deu bang nhau
#     (OHE: d(overcast, rainy) = d(overcast, sunny) = sqrt(2))
#   - Phu hop hon cho K-Means vi dam bao tinh doi xung
#
# NHUOC DIEM:
#   - Tang so chieu du lieu (4 thuoc tinh => 10 chieu sau OHE)
#   - Co the gap van de "loi nguyen chieu cao" (curse of dimensionality)
#     voi du lieu co nhieu gia tri phan biet
# ---------------------------------------------------------------------------
def create_ohe_encoding_table(data, cols):
    """
    Tao bang ma hoa One-Hot Encoding cho cac cot categorical.
    Tra ve:
      - encoding_table: dict {chi_so_cot: {gia_tri: vector_ohe}}
      - ohe_col_names: danh sach ten cac cot sau OHE (de hien thi)
    Vi du: Outlook co 3 gia tri => tao 3 cot: Outlook_overcast, Outlook_rainy, Outlook_sunny
    """
    encoding_table = {}
    ohe_col_names = []  # Ten cac cot sau OHE (de in bang)

    for col in cols:
        # Thu thap tat ca gia tri duy nhat cua cot
        unique_values = []
        for row in data:
            if row[col] not in unique_values:
                unique_values.append(row[col])
        unique_values.sort()  # Sap xep de nhat quan

        num_values = len(unique_values)

        # Tao vector OHE cho moi gia tri
        # Vi du: 3 gia tri => overcast=[1,0,0], rainy=[0,1,0], sunny=[0,0,1]
        encoding_table[col] = {}
        for i, value in enumerate(unique_values):
            vector = [0] * num_values       # Tao vector toan 0
            vector[i] = 1                    # Dat vi tri thu i = 1
            encoding_table[col][value] = vector

        # Luu ten cot OHE
        for value in unique_values:
            ohe_col_names.append(f"{value}")

    return encoding_table, ohe_col_names


def encode_data_ohe(data, encoding_table, cols):
    """
    Chuyen du lieu categorical thanh du lieu so bang One-Hot Encoding.
    Moi dong du lieu duoc chuyen thanh 1 vector nhi phan dai.
    Tra ve: list cac vector so (list of list)
    """
    numeric_data = []
    for row in data:
        vector = []
        for col in cols:
            value = row[col]
            # Noi (concatenate) vector OHE cua tung thuoc tinh
            vector.extend(encoding_table[col][value])
        numeric_data.append(vector)
    return numeric_data


# ---------------------------------------------------------------------------
# TINH KHOANG CACH EUCLID
# ---------------------------------------------------------------------------
# Cong thuc: d(x, y) = sqrt( sum( (xi - yi)^2 ) )
# Vi du: x = [0, 1, 0, 1], y = [1, 2, 1, 0]
#   d = sqrt( (0-1)^2 + (1-2)^2 + (0-1)^2 + (1-0)^2 )
#   d = sqrt( 1 + 1 + 1 + 1 ) = sqrt(4) = 2.0
# ---------------------------------------------------------------------------
def euclidean_distance(point_a, point_b):
    """
    Tinh khoang cach Euclid giua 2 diem (2 vector so).
    """
    total = 0
    for i in range(len(point_a)):
        total += (point_a[i] - point_b[i]) ** 2
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# KHOI TAO TAM CUM BAN DAU
# ---------------------------------------------------------------------------
# Cach 1: Chon K diem dau tien trong du lieu (don gian, de hieu)
# Cach 2: Chon ngau nhien K diem (thuc te dung cach nay)
# O day dung Cach 1 de ket qua co dinh, de kiem tra khi thi
# ---------------------------------------------------------------------------
def initialize_centroids(numeric_data, k):
    """
    Chon K diem dau tien lam tam cum ban dau.
    Tra ve: list cac tam cum (moi tam la 1 list so)
    """
    centroids = []
    for i in range(k):
        # Sao chep diem (tranh tham chieu)
        centroids.append(numeric_data[i][:])
    return centroids


# ---------------------------------------------------------------------------
# GAN MOI DIEM VAO CUM GAN NHAT
# ---------------------------------------------------------------------------
# Voi moi diem du lieu, tinh khoang cach den tat ca K tam cum,
# roi gan diem vao cum co tam gan nhat.
# ---------------------------------------------------------------------------
def assign_clusters(numeric_data, centroids):
    """
    Gan moi diem du lieu vao cum co tam cum gan nhat.
    Tra ve: list cac chi so cum (0, 1, ..., K-1) tuong ung voi tung diem
    """
    cluster_assignments = []  # Chi so cum cua tung diem
    for point in numeric_data:
        min_dist = float('inf')  # Vo cuc
        closest_cluster = 0

        for j, centroid in enumerate(centroids):
            dist = euclidean_distance(point, centroid)
            if dist < min_dist:
                min_dist = dist
                closest_cluster = j

        cluster_assignments.append(closest_cluster)

    return cluster_assignments


# ---------------------------------------------------------------------------
# CAP NHAT TAM CUM (TINH TRUNG BINH)
# ---------------------------------------------------------------------------
# Tam cum moi = trung binh cong cac toa do cua tat ca diem trong cum
# Vi du: Cum 1 co cac diem [0,1,0], [2,1,1], [1,0,1]
#   Tam moi = [(0+2+1)/3, (1+1+0)/3, (0+1+1)/3] = [1.0, 0.67, 0.67]
# ---------------------------------------------------------------------------
def update_centroids(numeric_data, cluster_assignments, k, num_dims):
    """
    Tinh tam cum moi = trung binh cac diem trong moi cum.
    Tra ve: list tam cum moi
    """
    new_centroids = []
    for j in range(k):
        # Lay tat ca diem thuoc cum j
        cluster_points = [numeric_data[i] for i in range(len(numeric_data)) if cluster_assignments[i] == j]

        if len(cluster_points) == 0:
            # Cum rong: giu nguyen tam cu (truong hop hiem gap)
            new_centroids.append([0.0] * num_dims)
        else:
            # Tinh trung binh tung chieu
            centroid = []
            for d in range(num_dims):
                mean_val = sum(point[d] for point in cluster_points) / len(cluster_points)
                centroid.append(round(mean_val, 4))
            new_centroids.append(centroid)

    return new_centroids


# ---------------------------------------------------------------------------
# KIEM TRA HOI TU
# ---------------------------------------------------------------------------
# Hoi tu khi tam cum khong thay doi (hoac thay doi rat nho) giua 2 vong lap
# ---------------------------------------------------------------------------
def has_converged(old_centroids, new_centroids, threshold=0.0001):
    """
    Kiem tra xem thuat toan da hoi tu chua.
    Hoi tu khi tong khoang cach dich chuyen cua cac tam < nguong.
    """
    total_shift = 0
    for i in range(len(old_centroids)):
        total_shift += euclidean_distance(old_centroids[i], new_centroids[i])
    return total_shift < threshold


# ---------------------------------------------------------------------------
# THUAT TOAN K-MEANS CHINH
# ---------------------------------------------------------------------------
def kmeans_algorithm(numeric_data, k, max_iterations=20):
    """
    Chay thuat toan K-Means.
    Tra ve: phan_cum (list chi so cum), tam_cum (list cac tam), lich_su (log)
    """
    num_dims = len(numeric_data[0])
    history = []  # Luu lich su moi vong lap

    # Buoc 1: Khoi tao tam cum
    centroids = initialize_centroids(numeric_data, k)

    for iteration in range(1, max_iterations + 1):
        # Buoc 2: Gan moi diem vao cum gan nhat
        cluster_assignments = assign_clusters(numeric_data, centroids)

        # Buoc 3: Cap nhat tam cum
        new_centroids = update_centroids(numeric_data, cluster_assignments, k, num_dims)

        # Luu lich su
        history.append({
            'vong': iteration,
            'tam_cum': [t[:] for t in centroids],
            'tam_moi': [t[:] for t in new_centroids],
            'phan_cum': cluster_assignments[:]
        })

        # Buoc 4: Kiem tra hoi tu
        if has_converged(centroids, new_centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    return cluster_assignments, centroids, history


# ===========================================================================
# PHAN CHINH: CHAY CHUONG TRINH
# ===========================================================================
def main():
    print("=" * 70)
    print("  THUAT TOAN K-MEANS CLUSTERING - PHAN CUM DU LIEU GOLF")
    print("=" * 70)

    # Doc du lieu
    file_path = "golf_df.csv"
    header, data = read_data(file_path)
    label_col = len(header) - 1

    # --- In bang du lieu goc ---
    print("\n[BANG DU LIEU GOC]")
    print("-" * 55)
    header_fmt = "{:<6} {:<12} {:<14} {:<10} {:<8} {:<6}"
    print(header_fmt.format("STT", *header))
    print("-" * 55)
    for i, row in enumerate(data, 1):
        print(header_fmt.format(i, *row))
    print("-" * 55)

    # --- Ma hoa du lieu ---
    # Chi dung cac thuoc tinh (khong dung cot nhan Play)
    cols = [i for i in range(len(header)) if i != label_col]
    encoding_table, ohe_col_names = create_ohe_encoding_table(data, cols)

    print("\n" + "=" * 70)
    print("[BUOC 0] MA HOA DU LIEU CATEGORICAL => SO (One-Hot Encoding)")
    print("-" * 70)
    print("\n  One-Hot Encoding: moi gia tri => 1 vector nhi phan,")
    print("  chi co dung 1 vi tri = 1, con lai = 0.")
    print("  => Khong tao quan he thu tu gia giua cac gia tri.")

    for col, encoding in encoding_table.items():
        col_name = header[col]
        num_values = len(encoding)
        print(f"\n  {col_name} ({num_values} gia tri => {num_values} cot OHE):")
        for value, vector in encoding.items():
            print(f"    {value:<12} => {vector}")

    numeric_data = encode_data_ohe(data, encoding_table, cols)
    num_dims = len(numeric_data[0])

    print(f"\n  Du lieu sau khi ma hoa ({num_dims} chieu):")
    print(f"  Ten cot OHE: {ohe_col_names}")
    header2 = "{:<6}" + "{:<4}" * num_dims
    print("  " + header2.format("STT", *ohe_col_names[:num_dims]))
    print("  " + "-" * (6 + 4 * num_dims))
    for i, vector in enumerate(numeric_data, 1):
        print("  " + header2.format(i, *vector))

    # --- Chay K-Means ---
    num_clusters = 2  # So cum (2 cum tuong ung voi yes/no, co the thay doi)
    print("\n" + "=" * 70)
    print(f"[K-MEANS] SO CUM K = {num_clusters}")
    print("=" * 70)

    # Buoc 1: Khoi tao tam cum
    print(f"\n[BUOC 1] KHOI TAO TAM CUM BAN DAU")
    print(f"  (Chon {num_clusters} diem dau tien lam tam cum)")
    for i in range(num_clusters):
        print(f"  Tam cum {i+1} (C{i+1}): {numeric_data[i]}")
        # Hien thi du lieu goc tuong ung
        original_row = [data[i][c] for c in cols]
        print(f"    (Tuong ung mau {i+1}: {original_row})")

    # Chay thuat toan
    cluster_assignments, final_centroids, history = kmeans_algorithm(numeric_data, num_clusters)

    # --- In chi tiet tung vong lap ---
    for log_entry in history:
        iteration = log_entry['vong']
        print(f"\n{'=' * 70}")
        print(f"[VONG LAP {iteration}]")
        print("-" * 55)

        # In bang khoang cach va phan cum
        print(f"\n  [BUOC 2] TINH KHOANG CACH VA GAN CUM:")
        header3 = "  {:<6}" + "{:<16}" * num_clusters + "{:<10}"
        centroid_names = [f"d(x,C{j+1})" for j in range(num_clusters)]
        print(header3.format("STT", *centroid_names, "Cum"))
        print("  " + "-" * (6 + 16 * num_clusters + 10))

        for i, point in enumerate(numeric_data):
            distances = []
            for j in range(num_clusters):
                dist = euclidean_distance(point, log_entry['tam_cum'][j])
                distances.append(f"{dist:.4f}")
            cluster_idx = log_entry['phan_cum'][i] + 1  # Hien thi tu 1
            row_vals = [i + 1] + distances + [f"C{cluster_idx}"]
            print(("  {:<6}" + "{:<16}" * num_clusters + "{:<10}").format(*row_vals))

        # In cac diem trong moi cum
        print(f"\n  Ket qua phan cum:")
        for j in range(num_clusters):
            cluster_point_indices = [i + 1 for i in range(len(numeric_data)) if log_entry['phan_cum'][i] == j]
            print(f"    Cum {j+1} (C{j+1}): Mau {cluster_point_indices} ({len(cluster_point_indices)} diem)")

        # In tam cum moi
        print(f"\n  [BUOC 3] CAP NHAT TAM CUM (Trung binh cac diem trong cum):")
        for j in range(num_clusters):
            cluster_points = [numeric_data[i] for i in range(len(numeric_data)) if log_entry['phan_cum'][i] == j]
            print(f"    Tam cum {j+1} cu:  {log_entry['tam_cum'][j]}")
            print(f"    Tam cum {j+1} moi: {log_entry['tam_moi'][j]}")

            # Chi tiet tinh trung binh cho moi chieu
            if len(cluster_points) > 0:
                details = []
                for d in range(len(cluster_points[0])):
                    dim_values = [p[d] for p in cluster_points]
                    mean_val = sum(dim_values) / len(dim_values)
                    details.append(f"({'+'.join(str(v) for v in dim_values)})/{len(dim_values)}={mean_val:.4f}")
                print(f"      Tinh: [{', '.join(details)}]")
            print()

        # Kiem tra hoi tu
        if has_converged(log_entry['tam_cum'], log_entry['tam_moi']):
            print(f"  => HOI TU! Tam cum khong thay doi.")
        else:
            shift = sum(
                euclidean_distance(log_entry['tam_cum'][j], log_entry['tam_moi'][j])
                for j in range(num_clusters)
            )
            print(f"  Tong dich chuyen tam cum: {shift:.4f} (chua hoi tu)")

    # --- Ket qua cuoi cung ---
    print("\n" + "=" * 70)
    print("[KET QUA CUOI CUNG]")
    print("-" * 55)
    print(f"  So vong lap: {len(history)}")
    print(f"  So cum K = {num_clusters}")
    print()

    # In tam cum cuoi
    for j in range(num_clusters):
        print(f"  Tam cum {j+1}: {final_centroids[j]}")

    # In ket qua phan cum
    print()
    for j in range(num_clusters):
        print(f"  Cum {j+1}:")
        for i in range(len(data)):
            if cluster_assignments[i] == j:
                original_row = data[i]
                print(f"    Mau {i+1:>2}: {', '.join(original_row[:label_col])} | Play={original_row[label_col]}")

    # --- So sanh voi nhan thuc te ---
    print("\n" + "=" * 70)
    print("[SO SANH VOI NHAN THUC TE (Play)]")
    print("-" * 55)
    print("  (K-Means la hoc KHONG GIAM SAT, khong biet nhan thuc te.")
    print("   Phan nay chi de so sanh tham khao.)")
    print()

    for j in range(num_clusters):
        label_counts = {}
        for i in range(len(data)):
            if cluster_assignments[i] == j:
                label = data[i][label_col]
                label_counts[label] = label_counts.get(label, 0) + 1

        total = sum(label_counts.values())
        print(f"  Cum {j+1} ({total} mau):")
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            print(f"    Play={label}: {count} mau ({percentage:.1f}%)")

    print("\n" + "=" * 70)
    print("  LUU Y: Du lieu categorical da duoc ma hoa bang One-Hot Encoding.")
    print("  OHE dam bao khoang cach giua cac gia tri khac nhau la nhu nhau,")
    print("  phu hop hon Label Encoding cho K-Means.")
    print("  Tuy nhien, so chieu tang len (4 thuoc tinh => " + str(num_dims) + " chieu).")
    print("=" * 70)

if __name__ == "__main__":
    main()
