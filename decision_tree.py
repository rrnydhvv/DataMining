# ===========================================================================
# THUẬT TOÁN DECISION TREE (CÂY QUYẾT ĐỊNH) - PHƯƠNG PHÁP ID3
# ===========================================================================

import csv
import math
from typing import List, Dict, Tuple, Any, Union

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
# TÍNH ENTROPY
# ---------------------------------------------------------------------------
def calculate_entropy(data: List[List[str]], label_col: int) -> float:
    """Tính Entropy của tập dữ liệu dựa trên cột nhãn."""
    #đếm số lượng mẫu
    total: int = len(data)
    if total == 0:
        return 0.0

    #khởi tại biến đểm số lượng từng lớp
    class_counts: Dict[str, int] = {}
    for row in data:
        #lấy nhãn của mẫu (yes/no)
        label = row[label_col]
        class_counts[label] = class_counts.get(label, 0) + 1

    entropy: float = 0.0
    for label, count in class_counts.items():
        p: float = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


# ---------------------------------------------------------------------------
# TÍNH INFORMATION GAIN
# ---------------------------------------------------------------------------
def calculate_information_gain(data: List[List[str]], feature_col: int, label_col: int) -> Tuple[float, Dict[str, Any], float]:
    """Tính Information Gain khi chia dữ liệu theo feature_col."""

    #đếm tổng số mẫu
    total: int = len(data)

    #tính entropy gốc của tập dữ liệu
    base_entropy: float = calculate_entropy(data, label_col)

    #tạo dict chứa các tập con
    subsets: Dict[str, List[List[str]]] = {}
    for row in data:
        #gom các dòng dữ liệu theo giá trị của thuộc tính feature_col 
        value = row[feature_col]
        #nếu giá trị chưa có trong dict, khởi tạo danh sách rỗng
        if value not in subsets:
            subsets[value] = []
        subsets[value].append(row)

    subset_entropy: float = 0.0
    subset_details: Dict[str, Any] = {}
    for value, subset in subsets.items(): #trả về giá trị và tập con tương ứng
        weight: float = len(subset) / total
        entropy_val: float = calculate_entropy(subset, label_col) #tính entropy của tập con
        subset_entropy += weight * entropy_val

        #khởi tạo bộ đếm lớp
        counts: Dict[str, int] = {} 
        for row in subset: #đếm số lượng từng nhãn, ví dụ sunny có bao nhiêu yes, bao nhiêu no
            n = row[label_col]
            counts[n] = counts.get(n, 0) + 1
        subset_details[value] = {
            'so_mau': len(subset),
            'entropy': entropy_val,
            'trong_so': weight,
            'dem_lop': counts
        }

    gain: float = base_entropy - subset_entropy
    return gain, subset_details, base_entropy


# ---------------------------------------------------------------------------
# XÂY DỰNG CÂY QUYẾT ĐỊNH
# ---------------------------------------------------------------------------
def build_tree(data: List[List[str]], header: List[str], remaining_cols: List[int], label_col: int, depth: int = 0) -> Union[str, Dict[str, Any], None]:
    """Xây dựng cây quyết định bằng thuật toán ID3 (đệ quy)."""
    if len(data) == 0:
        return None

    unique_labels = set(row[label_col] for row in data) #lấy tất cả các nhãn duy nhất trong tập dữ liệu
    if len(unique_labels) == 1:
        return list(unique_labels)[0]

    if len(remaining_cols) == 0:
        counts: Dict[str, int] = {}
        for row in data:
            n = row[label_col]
            counts[n] = counts.get(n, 0) + 1
        return max(counts, key=counts.get) # type: ignore

    best_gain: float = -1.0
    best_col: int = -1

    for col in remaining_cols:
        gain, _, _ = calculate_information_gain(data, col, label_col) #tính gain cho từng cột còn lại
        if gain > best_gain:
            best_gain = gain
            best_col = col

    feature_name = header[best_col]
    subsets: Dict[str, List[List[str]]] = {} #tạo dict chứa các tập con sau khi chia theo cột tốt nhất
    for row in data: #chia dữ liệu thành các nhóm theo thuộc tính được chọn
        value = row[best_col]
        if value not in subsets:
            subsets[value] = []
        subsets[value].append(row)

    new_cols = [c for c in remaining_cols if c != best_col] #loại bỏ cột đã chọn khỏi danh sách cột còn lại
    branches: Dict[str, Any] = {}
    #duyệt qua các nhánh của cột tốt nhất và xây dựng cây con cho mỗi nhánh
    for value, subset in subsets.items():
        branches[value] = build_tree(subset, header, new_cols, label_col, depth + 1)

    return {
        'thuoc_tinh': feature_name,
        'cot': best_col,
        'nhanh': branches
    }


# ---------------------------------------------------------------------------
# IN CÂY QUYẾT ĐỊNH
# ---------------------------------------------------------------------------
def print_tree(tree: Union[str, Dict[str, Any], None], prefix: str = "", is_last: bool = True) -> None:
    """In cây quyết định theo dạng cây thư mục (tree view)."""
    if tree is None:
        return
    if isinstance(tree, str):
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
# DỰ ĐOÁN MẪU MỚI
# ---------------------------------------------------------------------------
def predict(tree: Union[str, Dict[str, Any], None], sample: Dict[str, str], header: List[str]) -> str:
    """Duyệt cây từ gốc xuống lá để dự đoán lớp."""
    if tree is None:
        return "khong xac dinh"
    if isinstance(tree, str):
        return tree

    feature_name = tree['thuoc_tinh']
    sample_value = sample.get(feature_name, "").lower().strip()

    if sample_value in tree['nhanh']:
        return predict(tree['nhanh'][sample_value], sample, header)
    else:
        return "khong xac dinh"


# ---------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------
def main() -> None:
    print("--- THUAT TOAN DECISION TREE (ID3) ---")
    file_path = "golf_df.csv"
    header, data = read_data(file_path)
    label_col = len(header) - 1

    print(f"[Du lieu] Tong so mau: {len(data)}")

    base_entropy = calculate_entropy(data, label_col)
    print(f"\n[Buoc 1] Entropy goc toan bo tap du lieu: {base_entropy:.4f}")

    print("\n[Buoc 2] Information Gain cho tung thuoc tinh:")
    feature_cols = [i for i in range(len(header)) if i != label_col]
    gains: Dict[str, float] = {}

    for col in feature_cols:
        feature_name = header[col]
        gain, subset_details, _ = calculate_information_gain(data, col, label_col)
        gains[feature_name] = gain
        print(f"  Gain({feature_name}) = {gain:.4f}")

    best_feature = max(gains, key=gains.get) # type: ignore
    print(f"\n[Buoc 3] Chon thuoc tinh lam nut goc: {best_feature}")

    print("\n[Buoc 4] Cay quyet dinh:")
    tree = build_tree(data, header, feature_cols, label_col)
    print(f"  [{best_feature}]")
    print_tree(tree, prefix="  ")

    new_sample = {
        "Outlook": "sunny",
        "Temperature": "cool",
        "Humidity": "high",
        "Windy": "true"
    }

    print(f"\n[Buoc 5] Du doan mau moi: {new_sample}")
    
    result = predict(tree, new_sample, header)
    print(f"  => KET QUA: Play = {result.upper()}")

if __name__ == "__main__":
    main()