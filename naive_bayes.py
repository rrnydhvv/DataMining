# ===========================================================================
# THUẬT TOÁN NAIVE BAYES - PHÂN LOẠI (Classification)
# ===========================================================================

import csv
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
# XÁC SUẤT TIÊN NGHIỆM P(C)
# ---------------------------------------------------------------------------
def calculate_prior_probability(data: List[List[str]], label_col: int) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Tính xác suất tiên nghiệm P(C)."""
    total_samples: int = len(data)
    class_counts: Dict[str, int] = {}

    for row in data:
        label = row[label_col]#lấy giá trị của label trong 1 hàng
        class_counts[label] = class_counts.get(label, 0) + 1

    prior_probs: Dict[str, float] = {}
    for cls, count in class_counts.items():
        prior_probs[cls] = count / total_samples

    return prior_probs, class_counts


# ---------------------------------------------------------------------------
# XÁC SUẤT CÓ ĐIỀU KIỆN P(xi|C)
# ---------------------------------------------------------------------------
def calculate_conditional_probability(data: List[List[str]], feature_col: int, label_col: int) -> Dict[str, Dict[str, float]]:
    """Tính xác suất có điều kiện P(xi|C) kèm Laplace smoothing cơ bản nếu cần (sẽ xử lý kỹ khi dự đoán)."""
    counts: Dict[str, Dict[str, int]] = {} #đếm lớp nào có bao nhiêu thuộc tính
    class_counts: Dict[str, int] = {} #đếm số mẫu mối lớp

    for row in data:
        label = row[label_col] #lấy giá trị label (yes/no)
        value = row[feature_col] #lấy giá trị thuộc tính (sunny, overcast, rainy)

        if label not in counts:
            counts[label] = {}
            class_counts[label] = 0

        class_counts[label] += 1 #tăng số lượng mẫu của lớp
        counts[label][value] = counts[label].get(value, 0) + 1 #đếm số lần xuất hiện của giá trị thuộc tính trong lớp

    cond_probs: Dict[str, Dict[str, float]] = {}
    for cls in counts: #duyệt qua từng lớp (yes/no)
        cond_probs[cls] = {}
        for value in counts[cls]: #duyệt qua từng giá trị thuộc tính (sunny, overcast, rainy)
            cond_probs[cls][value] = counts[cls][value] / class_counts[cls]

    return cond_probs


# ---------------------------------------------------------------------------
# DỰ ĐOÁN
# ---------------------------------------------------------------------------
def predict_sample(new_sample: Dict[str, str], data: List[List[str]], header: List[str], label_col: int) -> Tuple[str, Dict[str, float], Dict[str, Any], Dict[str, Dict[str, Dict[str, float]]], Dict[str, float]]:
    """Dự đoán lớp bằng Naive Bayes với Laplace Smoothing."""
    prior_probs, class_counts = calculate_prior_probability(data, label_col) #tính xác suất tiên nghiệm 

    cond_prob_table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for i in range(len(header)): #duyệt tất cả các cột trong header
        if i == label_col: #bỏ qua cột label
            continue
        col_name = header[i]
        #tính xác suất có điều kiện P(xi|C) cho từng cột
        cond_prob_table[col_name] = calculate_conditional_probability(data, i, label_col)

    results: Dict[str, float] = {}
    details: Dict[str, Any] = {}

    for cls in prior_probs:
        product: float = prior_probs[cls]
        details[cls] = {
            'P(C)': prior_probs[cls],
            'cac_xs': {}
        }

        for feature_name, value in new_sample.items(): #duyệt từng thuộc tính trong mẫu mới
            value = value.lower().strip()
            cond_prob_dict = cond_prob_table[feature_name]

            if value in cond_prob_dict[cls]: #kiểm tra giá trị đã xuất hiện chưa
                prob_xi: float = cond_prob_dict[cls][value]
            else: #nếu chưa xuất hiện, áp dụng Laplace smoothing, dùng 1 xác suất nhỏ để tránh xác suất bằng 0
                num_values: int = len(cond_prob_dict[cls])
                prob_xi = 1 / (class_counts[cls] + num_values)

            details[cls]['cac_xs'][feature_name] = prob_xi
            product *= prob_xi

        results[cls] = product

    predicted_class: str = max(results, key=results.get) # type: ignore
    return predicted_class, results, details, cond_prob_table, prior_probs


# ---------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------
def main() -> None:
    print("--- THUAT TOAN NAIVE BAYES ---")
    file_path = "golf_df.csv"
    header, data = read_data(file_path)
    label_col = len(header) - 1

    print(f"[Du lieu] Tong so mau: {len(data)}")

    prior_probs, class_counts = calculate_prior_probability(data, label_col)
    print("\n[Buoc 1] Xac suat tien nghiem:")
    for cls, prob in prior_probs.items():
        print(f"  P({cls}) = {class_counts[cls]}/{len(data)} = {prob:.4f}")

    print("\n[Buoc 2] Xac suat co dieu kien (tom tat):")
    for i in range(len(header)):
        if i == label_col:
            continue
        col_name = header[i]
        cond_prob_dict = calculate_conditional_probability(data, i, label_col)
        
        all_values = set()
        for cls in cond_prob_dict:
            all_values.update(cond_prob_dict[cls].keys())
            
        print(f"  Thuoc tinh: {col_name}")
        for value in sorted(all_values):
            info = []
            for cls in prior_probs.keys():
                prob_val = cond_prob_dict.get(cls, {}).get(value, 0.0)
                count_val = int(prob_val * class_counts[cls])
                info.append(f"{cls}: {count_val}/{class_counts[cls]}={prob_val:.4f}")
            print(f"    {value}: {', '.join(info)}")

    new_sample = {
        "Outlook": "sunny",
        "Temperature": "cool",
        "Humidity": "high",
        "Windy": "true"
    }

    print("\n[Buoc 3] Du doan mau moi:")
    print(f"  Mau: {new_sample}")

    predicted_class, results, details, _, _ = predict_sample(new_sample, data, header, label_col)

    for cls in results:
        prior_prob = details[cls]['P(C)']
        calc_str = f"{prior_prob:.4f}"
        for feature_name, prob_val in details[cls]['cac_xs'].items():
            calc_str += f" * {prob_val:.4f}"
        print(f"  P({cls}|X) ty le voi: {calc_str} = {results[cls]:.6f}")

    print("\n[Buoc 4] Ket luan:")
    total_prob = sum(results.values())
    for cls, prob in results.items():
        percentage = (prob / total_prob) * 100 if total_prob > 0 else 0
        marker = " <== DU DOAN" if cls == predicted_class else ""
        print(f"  Lop '{cls}': {percentage:.2f}% (xs={prob:.6f}){marker}")

if __name__ == "__main__":
    main()
