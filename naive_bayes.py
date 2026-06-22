# ===========================================================================
# THUẬT TOÁN NAIVE BAYES - PHÂN LOẠI (Classification)
# ===========================================================================
# Mô tả:
#   Naive Bayes dựa trên định lý Bayes với giả định "ngây thơ" (naive) rằng
#   các thuộc tính độc lập với nhau khi biết nhãn lớp.
#
# Công thức Bayes:
#   P(C|X) = P(X|C) * P(C) / P(X)
#
#   Trong đó:
#     - C: lớp cần dự đoán (Play = yes / no)
#     - X: vector đặc trưng (Outlook, Temperature, Humidity, Windy)
#     - P(C): xác suất tiên nghiệm (prior) của lớp C
#     - P(X|C): xác suất có điều kiện (likelihood) của X khi biết C
#     - P(X): xác suất chung của X (hằng số, không cần tính khi so sánh)
#
# Giả định Naive (độc lập có điều kiện):
#   P(X|C) = P(x1|C) * P(x2|C) * ... * P(xn|C)
#
# Quy trình:
#   Bước 1: Tính xác suất tiên nghiệm P(C) cho từng lớp
#   Bước 2: Tính xác suất có điều kiện P(xi|C) cho từng thuộc tính
#   Bước 3: Với mẫu cần phân loại, tính P(C|X) cho từng lớp
#   Bước 4: Chọn lớp có P(C|X) lớn nhất
# ===========================================================================

import csv

# ---------------------------------------------------------------------------
# BUOC 0: Doc du lieu tu file CSV
# ---------------------------------------------------------------------------
def read_data(file_path):
    """
    Doc file CSV va tra ve:
      - header: danh sach ten cot
      - data: danh sach cac dong, moi dong la 1 danh sach gia tri
    """
    data = []
    header = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Dong dau tien la header
        for row in reader:
            if len(row) > 0:  # Bo qua dong trong
                # Chuan hoa: chuyen ve chu thuong, bo khoang trang thua
                clean_row = [value.strip().lower() for value in row]
                data.append(clean_row)
    return header, data


# ---------------------------------------------------------------------------
# BUOC 1: Tinh xac suat tien nghiem P(C) - Prior Probability
# ---------------------------------------------------------------------------
# Xac suat tien nghiem = So mau thuoc lop C / Tong so mau
# Vi du: P(Play=yes) = 9/14, P(Play=no) = 5/14
# ---------------------------------------------------------------------------
def calculate_prior_probability(data, label_col):
    """
    Tinh xac suat tien nghiem P(C) cho tung gia tri cua cot nhan (lop).
    Tra ve dict: {gia_tri_lop: xac_suat}
    Vi du: {'yes': 9/14, 'no': 5/14}
    """
    total_samples = len(data)
    class_counts = {}  # Dem so luong mau trong tung lop

    for row in data:
        label = row[label_col]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    # Tinh xac suat = so_mau_lop / tong_mau
    prior_probs = {}
    for cls, count in class_counts.items():
        prior_probs[cls] = count / total_samples

    return prior_probs, class_counts


# ---------------------------------------------------------------------------
# BUOC 2: Tinh xac suat co dieu kien P(xi|C) - Likelihood
# ---------------------------------------------------------------------------
# P(xi|C) = So mau co thuoc tinh xi VA thuoc lop C / So mau thuoc lop C
# Vi du: P(Outlook=sunny | Play=yes) = 2/9
# ---------------------------------------------------------------------------
def calculate_conditional_probability(data, feature_col, label_col):
    """
    Tinh xac suat co dieu kien P(xi|C) cho 1 thuoc tinh cu the.
    Tra ve dict long: {gia_tri_lop: {gia_tri_thuoc_tinh: xac_suat}}
    Vi du: {'yes': {'sunny': 2/9, 'overcast': 4/9, 'rainy': 3/9},
            'no':  {'sunny': 3/5, 'overcast': 0/5, 'rainy': 2/5}}
    """
    # Dem so mau theo (lop, gia_tri_thuoc_tinh)
    counts = {}       # {lop: {gia_tri: so_luong}}
    class_counts = {}   # {lop: tong_so_mau_cua_lop}

    for row in data:
        label = row[label_col]
        value = row[feature_col]

        if label not in counts:
            counts[label] = {}
            class_counts[label] = 0

        class_counts[label] += 1

        if value not in counts[label]:
            counts[label][value] = 0
        counts[label][value] += 1

    # Tinh xac suat P(xi|C) = counts[lop][gia_tri] / class_counts[lop]
    cond_probs = {}
    for cls in counts:
        cond_probs[cls] = {}
        for value in counts[cls]:
            cond_probs[cls][value] = counts[cls][value] / class_counts[cls]

    return cond_probs


# ---------------------------------------------------------------------------
# BUOC 3 & 4: Du doan - Tinh P(C|X) va chon lop co xs lon nhat
# ---------------------------------------------------------------------------
# P(C|X) ty le voi P(C) * P(x1|C) * P(x2|C) * ... * P(xn|C)
# (khong can chia cho P(X) vi P(X) la hang so chung cho moi lop)
#
# LAPLACE SMOOTHING (lam min Laplace):
#   Neu 1 gia tri thuoc tinh chua xuat hien trong lop C thi P(xi|C) = 0
#   => tich se bang 0 => sai ket qua
#   Cach xu ly: P(xi|C) = (so_mau + 1) / (tong_mau_lop + so_gia_tri_phan_biet)
#   Day la ky thuat pho bien de tranh xs = 0
# ---------------------------------------------------------------------------
def predict_sample(new_sample, data, header, label_col):
    """
    Du doan lop cho new_sample (dict: {ten_thuoc_tinh: gia_tri}).
    Tra ve: lop du doan, dict chi tiet xs tung lop
    """
    # Buoc 1: Tinh xs tien nghiem
    prior_probs, class_counts = calculate_prior_probability(data, label_col)

    # Buoc 2: Tinh xs co dieu kien cho tung thuoc tinh (tru cot nhan)
    # Luu tru: {ten_cot: {lop: {gia_tri: xs}}}
    cond_prob_table = {}
    for i in range(len(header)):
        if i == label_col:
            continue  # Bo qua cot nhan (cot Play)
        col_name = header[i]
        cond_prob_table[col_name] = calculate_conditional_probability(data, i, label_col)

    # Buoc 3: Tinh P(C|X) cho tung lop
    # P(C|X) ty le voi P(C) * tich P(xi|C)
    results = {}
    details = {}  # Luu chi tiet tung buoc de in ra

    for cls in prior_probs:
        # Bat dau voi P(C) - xs tien nghiem
        product = prior_probs[cls]
        details[cls] = {
            'P(C)': prior_probs[cls],
            'cac_xs': {}
        }

        # Nhan voi P(xi|C) cho tung thuoc tinh
        for feature_name, value in new_sample.items():
            value = value.lower().strip()
            cond_prob_dict = cond_prob_table[feature_name]

            if value in cond_prob_dict[cls]:
                prob_xi = cond_prob_dict[cls][value]
            else:
                # Laplace Smoothing: (0 + 1) / (tong + so_gia_tri_phan_biet)
                num_values = len(cond_prob_dict[cls])
                prob_xi = 1 / (class_counts[cls] + num_values)

            details[cls]['cac_xs'][feature_name] = prob_xi
            product *= prob_xi

        results[cls] = product

    # Buoc 4: Chon lop co xs lon nhat
    predicted_class = max(results, key=results.get)

    return predicted_class, results, details, cond_prob_table, prior_probs


# ===========================================================================
# PHAN CHINH: CHAY CHUONG TRINH
# ===========================================================================
def main():
    print("=" * 70)
    print("  THUAT TOAN NAIVE BAYES - PHAN LOAI CHOI GOLF")
    print("=" * 70)

    # Doc du lieu
    file_path = "golf_df.csv"
    header, data = read_data(file_path)

    # Xac dinh cot nhan (cot cuoi cung: Play)
    label_col = len(header) - 1
    label_name = header[label_col]

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

    # --- Buoc 1: Xac suat tien nghiem ---
    prior_probs, class_counts = calculate_prior_probability(data, label_col)
    print("\n" + "=" * 70)
    print("[BUOC 1] XAC SUAT TIEN NGHIEM P(Play)")
    print("-" * 40)
    for cls, prob in prior_probs.items():
        print(f"  P(Play={cls:<3}) = {class_counts[cls]}/{len(data)} = {prob:.4f}")

    # --- Buoc 2: Xac suat co dieu kien ---
    print("\n" + "=" * 70)
    print("[BUOC 2] XAC SUAT CO DIEU KIEN P(Xi | Play)")
    print("-" * 55)

    for i in range(len(header)):
        if i == label_col:
            continue
        col_name = header[i]
        cond_prob_dict = calculate_conditional_probability(data, i, label_col)

        print(f"\n  Thuoc tinh: {col_name}")
        print(f"  {'':.<4} {'Play=yes':>16} {'Play=no':>16}")

        # Thu thap tat ca gia tri cua thuoc tinh nay
        all_values = set()
        for cls in cond_prob_dict:
            all_values.update(cond_prob_dict[cls].keys())

        for value in sorted(all_values):
            prob_yes = cond_prob_dict.get('yes', {}).get(value, 0)
            prob_no = cond_prob_dict.get('no', {}).get(value, 0)

            # Tim so dem tuong ung
            count_yes = int(prob_yes * class_counts['yes'])
            count_no = int(prob_no * class_counts['no'])

            str_yes = f"{count_yes}/{class_counts['yes']} = {prob_yes:.4f}"
            str_no = f"{count_no}/{class_counts['no']} = {prob_no:.4f}"
            print(f"  {value:<14} {str_yes:>16} {str_no:>16}")

    # --- Buoc 3 & 4: Du doan mau moi ---
    # Mau can du doan (co the thay doi theo yeu cau giao vien)
    new_sample = {
        "Outlook": "sunny",
        "Temperature": "cool",
        "Humidity": "high",
        "Windy": "true"
    }

    print("\n" + "=" * 70)
    print("[BUOC 3] DU DOAN MAU MOI")
    print("-" * 40)
    print("  Mau can phan loai:")
    for feature_name, value in new_sample.items():
        print(f"    {feature_name} = {value}")

    predicted_class, results, details, _, _ = predict_sample(
        new_sample, data, header, label_col
    )

    print("\n  Tinh P(Play|X) cho tung lop:")
    print("-" * 55)

    for cls in results:
        print(f"\n  --- Play = {cls} ---")
        prior_prob = details[cls]['P(C)']
        print(f"  P(Play={cls}) = {prior_prob:.4f}")

        formula_str = f"P(Play={cls}|X) = P(Play={cls})"
        calc_str = f"{prior_prob:.4f}"

        for feature_name, prob_val in details[cls]['cac_xs'].items():
            print(f"  P({feature_name}={new_sample[feature_name]} | Play={cls}) = {prob_val:.4f}")
            formula_str += f" x P({feature_name}|{cls})"
            calc_str += f" x {prob_val:.4f}"

        print(f"\n  {formula_str}")
        print(f"  = {calc_str}")
        print(f"  = {results[cls]:.6f}")

    # --- Ket luan ---
    print("\n" + "=" * 70)
    print("[BUOC 4] KET LUAN")
    print("-" * 40)
    for cls, prob in results.items():
        marker = " (*)" if cls == predicted_class else ""
        print(f"  P(Play={cls:<3} | X) = {prob:.6f}{marker}")

    # Chuan hoa xac suat de hien thi phan tram
    total_prob = sum(results.values())
    print(f"\n  Chuan hoa xac suat (chia cho tong = {total_prob:.6f}):")
    for cls, prob in results.items():
        percentage = (prob / total_prob) * 100
        print(f"  P(Play={cls:<3} | X) = {percentage:.2f}%")

    print(f"\n  => KET QUA: Play = {predicted_class.upper()}")
    print(f"     (Chon lop co xac suat hau nghiem lon nhat)")
    print("=" * 70)

if __name__ == "__main__":
    main()
