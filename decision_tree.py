# ===========================================================================
# THUẬT TOÁN DECISION TREE (CÂY QUYẾT ĐỊNH) - PHƯƠNG PHÁP ID3
# ===========================================================================
# Mô tả:
#   Cây quyết định xây dựng cây phân loại bằng cách chọn thuộc tính tốt nhất
#   để chia dữ liệu tại mỗi nút. Thuộc tính tốt nhất là thuộc tính có
#   Information Gain (độ lợi thông tin) lớn nhất.
#
# Các khái niệm chính:
#
#   1. ENTROPY (Độ bất định / Độ hỗn loạn):
#      Entropy(S) = - sum( p_i * log2(p_i) )
#      Trong đó p_i là xác suất của lớp thứ i trong tập S
#      - Entropy = 0: tập dữ liệu thuần nhất (chỉ có 1 lớp)
#      - Entropy = 1: tập dữ liệu hỗn loạn nhất (các lớp cân bằng)
#
#   2. INFORMATION GAIN (Độ lợi thông tin):
#      Gain(S, A) = Entropy(S) - sum( |Sv|/|S| * Entropy(Sv) )
#      Trong đó:
#        - A: thuộc tính đang xét
#        - Sv: tập con của S ứng với giá trị v của thuộc tính A
#        - |Sv|/|S|: trọng số (tỷ lệ mẫu)
#      => Chọn thuộc tính A có Gain lớn nhất để chia
#
# Quy trình xây dựng cây (thuật toán ID3):
#   Bước 1: Tính Entropy của toàn bộ tập dữ liệu
#   Bước 2: Tính Information Gain cho từng thuộc tính
#   Bước 3: Chọn thuộc tính có Gain lớn nhất làm nút gốc
#   Bước 4: Chia dữ liệu theo các giá trị của thuộc tính đã chọn
#   Bước 5: Lặp lại Bước 1-4 cho từng nhánh (đệ quy)
#   Điều kiện dừng: Entropy = 0 (nút lá) hoặc hết thuộc tính
# ===========================================================================

import csv
import math

# ---------------------------------------------------------------------------
# BUOC 0: Doc du lieu tu file CSV
# ---------------------------------------------------------------------------
def read_data(file_path):
    """
    Doc file CSV va tra ve header (tieu_de) va data (list cac dong).
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
# TINH ENTROPY
# ---------------------------------------------------------------------------
# Cong thuc: Entropy(S) = - sum( p_i * log2(p_i) )
# Vi du: S co 9 yes, 5 no
#   p_yes = 9/14, p_no = 5/14
#   Entropy = -(9/14)*log2(9/14) - (5/14)*log2(5/14) = 0.9403
# ---------------------------------------------------------------------------
def calculate_entropy(data, label_col):
    """
    Tinh Entropy cua tap du lieu dua tren cot nhan (lop).
    Entropy do muc do hon loan / bat dinh cua du lieu.
    """
    total = len(data)
    if total == 0:
        return 0

    # Dem so luong tung lop
    class_counts = {}
    for row in data:
        label = row[label_col]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    # Ap dung cong thuc Entropy
    entropy = 0
    for label, count in class_counts.items():
        p = count / total           # Xac suat cua lop
        if p > 0:
            entropy -= p * math.log2(p)  # -p * log2(p)

    return entropy


# ---------------------------------------------------------------------------
# TINH INFORMATION GAIN
# ---------------------------------------------------------------------------
# Cong thuc: Gain(S, A) = Entropy(S) - sum( |Sv|/|S| * Entropy(Sv) )
#
# Y nghia: Gain do muc do giam Entropy (giam bat dinh) khi chia theo A
# Gain cang lon => thuoc tinh A cang tot de phan loai
# ---------------------------------------------------------------------------
def calculate_information_gain(data, feature_col, label_col):
    """
    Tinh Information Gain khi chia du lieu theo cot_thuoc_tinh (feature_col).
    Tra ve: gia tri Gain, dict cac tap con (de in chi tiet)
    """
    total = len(data)
    base_entropy = calculate_entropy(data, label_col)

    # Chia du lieu thanh cac tap con theo gia tri cua thuoc tinh
    subsets = {}  # {gia_tri: [cac dong thuoc gia tri do]}
    for row in data:
        value = row[feature_col]
        if value not in subsets:
            subsets[value] = []
        subsets[value].append(row)

    # Tinh Entropy trung binh co trong so cua cac tap con
    subset_entropy = 0
    subset_details = {}  # Luu chi tiet de in
    for value, subset in subsets.items():
        weight = len(subset) / total                    # |Sv| / |S|
        entropy_val = calculate_entropy(subset, label_col)       # Entropy(Sv)
        subset_entropy += weight * entropy_val           # Cong don

        # Luu chi tiet
        counts = {}
        for row in subset:
            n = row[label_col]
            counts[n] = counts.get(n, 0) + 1
        subset_details[value] = {
            'so_mau': len(subset),
            'entropy': entropy_val,
            'trong_so': weight,
            'dem_lop': counts
        }

    # Gain = Entropy(goc) - Entropy trung binh cac tap con
    gain = base_entropy - subset_entropy

    return gain, subset_details, base_entropy


# ---------------------------------------------------------------------------
# XAY DUNG CAY QUYET DINH (DE QUY)
# ---------------------------------------------------------------------------
def build_tree(data, header, remaining_cols, label_col, depth=0):
    """
    Xay dung cay quyet dinh bang thuat toan ID3 (de quy).
    Tra ve: dict bieu dien cay
      - Nut la: gia tri la nhan lop (vi du: 'yes')
      - Nut trong: dict {'thuoc_tinh': ten, 'nhanh': {gia_tri: cay_con}}
    """
    # --- Dieu kien dung de quy ---

    # Truong hop 1: Tat ca mau cung 1 lop => tra ve nhan do (nut la)
    unique_labels = set(row[label_col] for row in data)
    if len(unique_labels) == 1:
        return list(unique_labels)[0]

    # Truong hop 2: Het thuoc tinh de chia => tra ve lop da so (majority vote)
    if len(remaining_cols) == 0:
        counts = {}
        for row in data:
            n = row[label_col]
            counts[n] = counts.get(n, 0) + 1
        return max(counts, key=counts.get)

    # Truong hop 3: Khong con du lieu => tra ve None
    if len(data) == 0:
        return None

    # --- Tim thuoc tinh co Information Gain lon nhat ---
    best_gain = -1
    best_col = -1

    for col in remaining_cols:
        gain, _, _ = calculate_information_gain(data, col, label_col)
        if gain > best_gain:
            best_gain = gain
            best_col = col

    # Tao nut cay voi thuoc tinh tot nhat
    feature_name = header[best_col]

    # Chia du lieu theo gia tri cua thuoc tinh tot nhat
    subsets = {}
    for row in data:
        value = row[best_col]
        if value not in subsets:
            subsets[value] = []
        subsets[value].append(row)

    # Tao cac nhanh con (de quy)
    new_cols = [c for c in remaining_cols if c != best_col]
    branches = {}
    for value, subset in subsets.items():
        branches[value] = build_tree(subset, header, new_cols, label_col, depth + 1)

    return {
        'thuoc_tinh': feature_name,
        'cot': best_col,
        'nhanh': branches
    }


# ---------------------------------------------------------------------------
# IN CAY QUYET DINH (TRUC QUAN HOA TREN TERMINAL)
# ---------------------------------------------------------------------------
def print_tree(tree, prefix="", is_last=True):
    """
    In cay quyet dinh theo dang cay thu muc (tree view) tren terminal.
    """
    if isinstance(tree, str):
        # Nut la: in nhan lop
        print(f"=> [{tree.upper()}]")
        return

    feature_name = tree['thuoc_tinh']
    branches_list = list(tree['nhanh'].items())

    for i, (value, subtree) in enumerate(branches_list):
        is_last_branch = (i == len(branches_list) - 1)
        symbol = "`-- " if is_last_branch else "|-- "
        connector = "    " if is_last_branch else "|   "

        if isinstance(subtree, str):
            print(f"{prefix}{symbol}[{feature_name} = {value}] => [{subtree.upper()}]")
        else:
            print(f"{prefix}{symbol}[{feature_name} = {value}]")
            print_tree(subtree, prefix + connector, is_last_branch)


# ---------------------------------------------------------------------------
# DU DOAN MAU MOI BANG CAY QUYET DINH
# ---------------------------------------------------------------------------
def predict(tree, sample, header):
    """
    Duyet cay tu goc xuong la de du doan lop cho mau moi.
    Mau la dict: {ten_thuoc_tinh: gia_tri}
    """
    # Nut la => tra ve nhan lop
    if isinstance(tree, str):
        return tree

    feature_name = tree['thuoc_tinh']
    sample_value = sample.get(feature_name, "").lower().strip()

    # Tim nhanh tuong ung voi gia tri cua mau
    if sample_value in tree['nhanh']:
        return predict(tree['nhanh'][sample_value], sample, header)
    else:
        # Truong hop gia tri khong ton tai trong cay
        # => tra ve nhan phoi bien nhat (fallback)
        return "khong xac dinh"


# ===========================================================================
# PHAN CHINH: CHAY CHUONG TRINH
# ===========================================================================
def main():
    print("=" * 70)
    print("  THUAT TOAN DECISION TREE (ID3) - PHAN LOAI CHOI GOLF")
    print("=" * 70)

    # Doc du lieu
    file_path = "golf_df.csv"
    header, data = read_data(file_path)
    label_col = len(header) - 1  # Cot cuoi cung (Play)

    # --- In bang du lieu ---
    print("\n[BANG DU LIEU]")
    print("-" * 55)
    header_fmt = "{:<6} {:<12} {:<14} {:<10} {:<8} {:<6}"
    print(header_fmt.format("STT", *header))
    print("-" * 55)
    for i, row in enumerate(data, 1):
        print(header_fmt.format(i, *row))
    print("-" * 55)
    print(f"Tong so mau: {len(data)}")

    # --- Buoc 1: Tinh Entropy toan bo tap du lieu ---
    base_entropy = calculate_entropy(data, label_col)
    class_counts = {}
    for row in data:
        n = row[label_col]
        class_counts[n] = class_counts.get(n, 0) + 1

    print("\n" + "=" * 70)
    print("[BUOC 1] ENTROPY CUA TOAN BO TAP DU LIEU")
    print("-" * 55)
    print(f"  Tong mau: {len(data)}")
    for label, count in class_counts.items():
        print(f"  Play={label}: {count} mau (p = {count}/{len(data)} = {count/len(data):.4f})")

    formula_str = "  Entropy(S) = "
    parts = []
    for label, count in class_counts.items():
        p = count / len(data)
        parts.append(f"-({count}/{len(data)})*log2({count}/{len(data)})")
    formula_str += " + ".join(parts)
    print(formula_str)
    print(f"  Entropy(S) = {base_entropy:.4f}")

    # --- Buoc 2: Tinh Information Gain cho tung thuoc tinh ---
    print("\n" + "=" * 70)
    print("[BUOC 2] INFORMATION GAIN CHO TUNG THUOC TINH")
    print("-" * 55)

    feature_cols = [i for i in range(len(header)) if i != label_col]
    gains = {}

    for col in feature_cols:
        feature_name = header[col]
        gain, subset_details, entropy_s = calculate_information_gain(data, col, label_col)
        gains[feature_name] = gain

        print(f"\n  --- Thuoc tinh: {feature_name} ---")

        # In chi tiet tung tap con
        avg_entropy_strs = []
        for value, info in subset_details.items():
            count_str = ", ".join(f"{k}={v}" for k, v in info['dem_lop'].items())
            print(f"  {feature_name}={value}: {info['so_mau']} mau ({count_str})")
            print(f"    Entropy({value}) = {info['entropy']:.4f}")
            avg_entropy_strs.append(
                f"({info['so_mau']}/{len(data)})*{info['entropy']:.4f}"
            )

        print(f"\n  Gain({feature_name}) = Entropy(S) - [{' + '.join(avg_entropy_strs)}]")
        print(f"  Gain({feature_name}) = {entropy_s:.4f} - {entropy_s - gain:.4f} = {gain:.4f}")

    # --- Buoc 3: Chon thuoc tinh co Gain lon nhat ---
    best_feature = max(gains, key=gains.get)
    print("\n" + "=" * 70)
    print("[BUOC 3] SO SANH INFORMATION GAIN")
    print("-" * 40)
    for feature_name, gain in gains.items():
        marker = " <<< MAX" if feature_name == best_feature else ""
        print(f"  Gain({feature_name:<14}) = {gain:.4f}{marker}")
    print(f"\n  => Chon thuoc tinh '{best_feature}' lam nut goc (Gain lon nhat)")

    # --- Buoc 4: Xay dung cay quyet dinh ---
    print("\n" + "=" * 70)
    print("[BUOC 4] XAY DUNG CAY QUYET DINH (DE QUY)")
    print("-" * 40)

    tree = build_tree(data, header, feature_cols, label_col)

    print(f"\n  Nut goc: [{best_feature}]")
    print_tree(tree, prefix="  ")

    # --- Buoc 5: Du doan mau moi ---
    # Mau can du doan (co the thay doi theo yeu cau giao vien)
    new_sample = {
        "Outlook": "sunny",
        "Temperature": "cool",
        "Humidity": "high",
        "Windy": "true"
    }

    print("\n" + "=" * 70)
    print("[BUOC 5] DU DOAN MAU MOI")
    print("-" * 40)
    print("  Mau can phan loai:")
    for feature_name, value in new_sample.items():
        print(f"    {feature_name} = {value}")

    # Truy vet duong di tren cay
    print("\n  Qua trinh duyet cay:")
    current_tree = tree
    step = 1
    while isinstance(current_tree, dict):
        feature_name = current_tree['thuoc_tinh']
        value = new_sample.get(feature_name, "").lower()
        print(f"    Buoc {step}: {feature_name} = {value} --> di theo nhanh '{value}'")
        current_tree = current_tree['nhanh'].get(value, "khong xac dinh")
        step += 1

    result = predict(tree, new_sample, header)
    print(f"\n  => KET QUA: Play = {result.upper()}")
    print("=" * 70)
    
    # Thêm mẫu được dự đoán vào lại file csv gốc
    # with open(file_path, mode='a', encoding='utf-8', newline='') as f:
    #     writer = csv.writer(f)
    #     new_row = [new_sample.get(name, "") for name in header[:-1]] + [result]
    #     writer.writerow(new_row)
if __name__ == "__main__":
    main()