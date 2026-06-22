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
def doc_du_lieu(duong_dan):
    """
    Doc file CSV va tra ve:
      - tieu_de: danh sach ten cot (header)
      - du_lieu: danh sach cac dong, moi dong la 1 danh sach gia tri
    """
    du_lieu = []
    tieu_de = []
    with open(duong_dan, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        tieu_de = next(reader)  # Dong dau tien la header
        for dong in reader:
            if len(dong) > 0:  # Bo qua dong trong
                # Chuan hoa: chuyen ve chu thuong, bo khoang trang thua
                dong_sach = [gia_tri.strip().lower() for gia_tri in dong]
                du_lieu.append(dong_sach)
    return tieu_de, du_lieu


# ---------------------------------------------------------------------------
# BUOC 1: Tinh xac suat tien nghiem P(C) - Prior Probability
# ---------------------------------------------------------------------------
# Xac suat tien nghiem = So mau thuoc lop C / Tong so mau
# Vi du: P(Play=yes) = 9/14, P(Play=no) = 5/14
# ---------------------------------------------------------------------------
def tinh_xac_suat_tien_nghiem(du_lieu, cot_nhan):
    """
    Tinh xac suat tien nghiem P(C) cho tung gia tri cua cot nhan (lop).
    Tra ve dict: {gia_tri_lop: xac_suat}
    Vi du: {'yes': 9/14, 'no': 5/14}
    """
    tong_mau = len(du_lieu)
    dem_lop = {}  # Dem so luong mau trong tung lop

    for dong in du_lieu:
        nhan = dong[cot_nhan]
        if nhan not in dem_lop:
            dem_lop[nhan] = 0
        dem_lop[nhan] += 1

    # Tinh xac suat = so_mau_lop / tong_mau
    xs_tien_nghiem = {}
    for lop, so_luong in dem_lop.items():
        xs_tien_nghiem[lop] = so_luong / tong_mau

    return xs_tien_nghiem, dem_lop


# ---------------------------------------------------------------------------
# BUOC 2: Tinh xac suat co dieu kien P(xi|C) - Likelihood
# ---------------------------------------------------------------------------
# P(xi|C) = So mau co thuoc tinh xi VA thuoc lop C / So mau thuoc lop C
# Vi du: P(Outlook=sunny | Play=yes) = 2/9
# ---------------------------------------------------------------------------
def tinh_xac_suat_dieu_kien(du_lieu, cot_thuoc_tinh, cot_nhan):
    """
    Tinh xac suat co dieu kien P(xi|C) cho 1 thuoc tinh cu the.
    Tra ve dict long: {gia_tri_lop: {gia_tri_thuoc_tinh: xac_suat}}
    Vi du: {'yes': {'sunny': 2/9, 'overcast': 4/9, 'rainy': 3/9},
            'no':  {'sunny': 3/5, 'overcast': 0/5, 'rainy': 2/5}}
    """
    # Dem so mau theo (lop, gia_tri_thuoc_tinh)
    dem = {}       # {lop: {gia_tri: so_luong}}
    dem_lop = {}   # {lop: tong_so_mau_cua_lop}

    for dong in du_lieu:
        nhan = dong[cot_nhan]
        gia_tri = dong[cot_thuoc_tinh]

        if nhan not in dem:
            dem[nhan] = {}
            dem_lop[nhan] = 0

        dem_lop[nhan] += 1

        if gia_tri not in dem[nhan]:
            dem[nhan][gia_tri] = 0
        dem[nhan][gia_tri] += 1

    # Tinh xac suat P(xi|C) = dem[lop][gia_tri] / dem_lop[lop]
    xs_dieu_kien = {}
    for lop in dem:
        xs_dieu_kien[lop] = {}
        for gia_tri in dem[lop]:
            xs_dieu_kien[lop][gia_tri] = dem[lop][gia_tri] / dem_lop[lop]

    return xs_dieu_kien


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
def du_doan(mau_moi, du_lieu, tieu_de, cot_nhan):
    """
    Du doan lop cho mau_moi (dict: {ten_thuoc_tinh: gia_tri}).
    Tra ve: lop du doan, dict chi tiet xs tung lop
    """
    # Buoc 1: Tinh xs tien nghiem
    xs_tien_nghiem, dem_lop = tinh_xac_suat_tien_nghiem(du_lieu, cot_nhan)

    # Buoc 2: Tinh xs co dieu kien cho tung thuoc tinh (tru cot nhan)
    # Luu tru: {ten_cot: {lop: {gia_tri: xs}}}
    bang_xs_dieu_kien = {}
    for i in range(len(tieu_de)):
        if i == cot_nhan:
            continue  # Bo qua cot nhan (cot Play)
        ten_cot = tieu_de[i]
        bang_xs_dieu_kien[ten_cot] = tinh_xac_suat_dieu_kien(du_lieu, i, cot_nhan)

    # Buoc 3: Tinh P(C|X) cho tung lop
    # P(C|X) ty le voi P(C) * tich P(xi|C)
    ket_qua = {}
    chi_tiet = {}  # Luu chi tiet tung buoc de in ra

    for lop in xs_tien_nghiem:
        # Bat dau voi P(C) - xs tien nghiem
        tich = xs_tien_nghiem[lop]
        chi_tiet[lop] = {
            'P(C)': xs_tien_nghiem[lop],
            'cac_xs': {}
        }

        # Nhan voi P(xi|C) cho tung thuoc tinh
        for ten_thuoc_tinh, gia_tri in mau_moi.items():
            gia_tri = gia_tri.lower().strip()
            xs_dk = bang_xs_dieu_kien[ten_thuoc_tinh]

            if gia_tri in xs_dk[lop]:
                xs_xi = xs_dk[lop][gia_tri]
            else:
                # Laplace Smoothing: (0 + 1) / (tong + so_gia_tri_phan_biet)
                so_gia_tri = len(xs_dk[lop])
                xs_xi = 1 / (dem_lop[lop] + so_gia_tri)

            chi_tiet[lop]['cac_xs'][ten_thuoc_tinh] = xs_xi
            tich *= xs_xi

        ket_qua[lop] = tich

    # Buoc 4: Chon lop co xs lon nhat
    lop_du_doan = max(ket_qua, key=ket_qua.get)

    return lop_du_doan, ket_qua, chi_tiet, bang_xs_dieu_kien, xs_tien_nghiem


# ===========================================================================
# PHAN CHINH: CHAY CHUONG TRINH
# ===========================================================================
def main():
    print("=" * 70)
    print("  THUAT TOAN NAIVE BAYES - PHAN LOAI CHOI GOLF")
    print("=" * 70)

    # Doc du lieu
    duong_dan = "golf_df.csv"
    tieu_de, du_lieu = doc_du_lieu(duong_dan)

    # Xac dinh cot nhan (cot cuoi cung: Play)
    cot_nhan = len(tieu_de) - 1
    ten_nhan = tieu_de[cot_nhan]

    # --- In bang du lieu ---
    print("\n[BANG DU LIEU]")
    print("-" * 55)
    header_fmt = "{:<6} {:<12} {:<14} {:<10} {:<8} {:<6}"
    print(header_fmt.format("STT", *tieu_de))
    print("-" * 55)
    for i, dong in enumerate(du_lieu, 1):
        print(header_fmt.format(i, *dong))
    print("-" * 55)
    print(f"Tong so mau: {len(du_lieu)}")

    # --- Buoc 1: Xac suat tien nghiem ---
    xs_tien_nghiem, dem_lop = tinh_xac_suat_tien_nghiem(du_lieu, cot_nhan)
    print("\n" + "=" * 70)
    print("[BUOC 1] XAC SUAT TIEN NGHIEM P(Play)")
    print("-" * 40)
    for lop, xs in xs_tien_nghiem.items():
        print(f"  P(Play={lop:<3}) = {dem_lop[lop]}/{len(du_lieu)} = {xs:.4f}")

    # --- Buoc 2: Xac suat co dieu kien ---
    print("\n" + "=" * 70)
    print("[BUOC 2] XAC SUAT CO DIEU KIEN P(Xi | Play)")
    print("-" * 55)

    for i in range(len(tieu_de)):
        if i == cot_nhan:
            continue
        ten_cot = tieu_de[i]
        xs_dk = tinh_xac_suat_dieu_kien(du_lieu, i, cot_nhan)

        print(f"\n  Thuoc tinh: {ten_cot}")
        print(f"  {'':.<4} {'Play=yes':>16} {'Play=no':>16}")

        # Thu thap tat ca gia tri cua thuoc tinh nay
        tat_ca_gia_tri = set()
        for lop in xs_dk:
            tat_ca_gia_tri.update(xs_dk[lop].keys())

        for gia_tri in sorted(tat_ca_gia_tri):
            xs_yes = xs_dk.get('yes', {}).get(gia_tri, 0)
            xs_no = xs_dk.get('no', {}).get(gia_tri, 0)

            # Tim so dem tuong ung
            dem_yes = int(xs_yes * dem_lop['yes'])
            dem_no = int(xs_no * dem_lop['no'])

            col_yes = f"{dem_yes}/{dem_lop['yes']} = {xs_yes:.4f}"
            col_no = f"{dem_no}/{dem_lop['no']} = {xs_no:.4f}"
            print(f"  {gia_tri:<14} {col_yes:>16} {col_no:>16}")

    # --- Buoc 3 & 4: Du doan mau moi ---
    # Mau can du doan (co the thay doi theo yeu cau giao vien)
    mau_moi = {
        "Outlook": "sunny",
        "Temperature": "cool",
        "Humidity": "high",
        "Windy": "true"
    }

    print("\n" + "=" * 70)
    print("[BUOC 3] DU DOAN MAU MOI")
    print("-" * 40)
    print("  Mau can phan loai:")
    for ten, gia_tri in mau_moi.items():
        print(f"    {ten} = {gia_tri}")

    lop_du_doan, ket_qua, chi_tiet, _, _ = du_doan(
        mau_moi, du_lieu, tieu_de, cot_nhan
    )

    print("\n  Tinh P(Play|X) cho tung lop:")
    print("-" * 55)

    for lop in ket_qua:
        print(f"\n  --- Play = {lop} ---")
        prior = chi_tiet[lop]['P(C)']
        print(f"  P(Play={lop}) = {prior:.4f}")

        cong_thuc = f"P(Play={lop}|X) = P(Play={lop})"
        phep_tinh = f"{prior:.4f}"

        for ten_tt, xs_val in chi_tiet[lop]['cac_xs'].items():
            print(f"  P({ten_tt}={mau_moi[ten_tt]} | Play={lop}) = {xs_val:.4f}")
            cong_thuc += f" x P({ten_tt}|{lop})"
            phep_tinh += f" x {xs_val:.4f}"

        print(f"\n  {cong_thuc}")
        print(f"  = {phep_tinh}")
        print(f"  = {ket_qua[lop]:.6f}")

    # --- Ket luan ---
    print("\n" + "=" * 70)
    print("[BUOC 4] KET LUAN")
    print("-" * 40)
    for lop, xs in ket_qua.items():
        dau = " (*)" if lop == lop_du_doan else ""
        print(f"  P(Play={lop:<3} | X) = {xs:.6f}{dau}")

    # Chuan hoa xac suat de hien thi phan tram
    tong_xs = sum(ket_qua.values())
    print(f"\n  Chuan hoa xac suat (chia cho tong = {tong_xs:.6f}):")
    for lop, xs in ket_qua.items():
        phan_tram = (xs / tong_xs) * 100
        print(f"  P(Play={lop:<3} | X) = {phan_tram:.2f}%")

    print(f"\n  => KET QUA: Play = {lop_du_doan.upper()}")
    print(f"     (Chon lop co xac suat hau nghiem lon nhat)")
    print("=" * 70)

if __name__ == "__main__":
    main()
