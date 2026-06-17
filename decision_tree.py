# ===========================================================================
# THUAT TOAN DECISION TREE (CAY QUYET DINH) - PHUONG PHAP ID3
# ===========================================================================
# Mo ta:
#   Cay quyet dinh xay dung cay phan loai bang cach chon thuoc tinh tot nhat
#   de chia du lieu tai moi nut. Thuoc tinh tot nhat la thuoc tinh co
#   Information Gain (do loi thong tin) lon nhat.
#
# Cac khai niem chinh:
#
#   1. ENTROPY (Do bat dinh / Do hon loan):
#      Entropy(S) = - sum( p_i * log2(p_i) )
#      Trong do p_i la xac suat cua lop thu i trong tap S
#      - Entropy = 0: tap du lieu thuan nhat (chi co 1 lop)
#      - Entropy = 1: tap du lieu hon loan nhat (cac lop can bang)
#
#   2. INFORMATION GAIN (Do loi thong tin):
#      Gain(S, A) = Entropy(S) - sum( |Sv|/|S| * Entropy(Sv) )
#      Trong do:
#        - A: thuoc tinh dang xet
#        - Sv: tap con cua S ung voi gia tri v cua thuoc tinh A
#        - |Sv|/|S|: trong so (ty le mau)
#      => Chon thuoc tinh A co Gain lon nhat de chia
#
# Quy trinh xay dung cay (thuat toan ID3):
#   Buoc 1: Tinh Entropy cua toan bo tap du lieu
#   Buoc 2: Tinh Information Gain cho tung thuoc tinh
#   Buoc 3: Chon thuoc tinh co Gain lon nhat lam nut goc
#   Buoc 4: Chia du lieu theo cac gia tri cua thuoc tinh da chon
#   Buoc 5: Lap lai Buoc 1-4 cho tung nhanh (de quy)
#   Dieu kien dung: Entropy = 0 (nut la) hoac het thuoc tinh
# ===========================================================================

import csv
import math

# ---------------------------------------------------------------------------
# BUOC 0: Doc du lieu tu file CSV
# ---------------------------------------------------------------------------
def doc_du_lieu(duong_dan):
    """
    Doc file CSV va tra ve tieu_de (header) va du_lieu (list cac dong).
    """
    du_lieu = []
    tieu_de = []
    with open(duong_dan, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        tieu_de = next(reader)
        for dong in reader:
            if len(dong) > 0:
                dong_sach = [gia_tri.strip().lower() for gia_tri in dong]
                du_lieu.append(dong_sach)
    return tieu_de, du_lieu


# ---------------------------------------------------------------------------
# TINH ENTROPY
# ---------------------------------------------------------------------------
# Cong thuc: Entropy(S) = - sum( p_i * log2(p_i) )
# Vi du: S co 9 yes, 5 no
#   p_yes = 9/14, p_no = 5/14
#   Entropy = -(9/14)*log2(9/14) - (5/14)*log2(5/14) = 0.9403
# ---------------------------------------------------------------------------
def tinh_entropy(du_lieu, cot_nhan):
    """
    Tinh Entropy cua tap du lieu dua tren cot nhan (lop).
    Entropy do muc do hon loan / bat dinh cua du lieu.
    """
    tong = len(du_lieu)
    if tong == 0:
        return 0

    # Dem so luong tung lop
    dem_lop = {}
    for dong in du_lieu:
        nhan = dong[cot_nhan]
        if nhan not in dem_lop:
            dem_lop[nhan] = 0
        dem_lop[nhan] += 1

    # Ap dung cong thuc Entropy
    entropy = 0
    for lop, so_luong in dem_lop.items():
        p = so_luong / tong           # Xac suat cua lop
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
def tinh_information_gain(du_lieu, cot_thuoc_tinh, cot_nhan):
    """
    Tinh Information Gain khi chia du lieu theo cot_thuoc_tinh.
    Tra ve: gia tri Gain, dict cac tap con (de in chi tiet)
    """
    tong = len(du_lieu)
    entropy_goc = tinh_entropy(du_lieu, cot_nhan)

    # Chia du lieu thanh cac tap con theo gia tri cua thuoc tinh
    tap_con = {}  # {gia_tri: [cac dong thuoc gia tri do]}
    for dong in du_lieu:
        gia_tri = dong[cot_thuoc_tinh]
        if gia_tri not in tap_con:
            tap_con[gia_tri] = []
        tap_con[gia_tri].append(dong)

    # Tinh Entropy trung binh co trong so cua cac tap con
    entropy_con = 0
    chi_tiet_con = {}  # Luu chi tiet de in
    for gia_tri, tap in tap_con.items():
        trong_so = len(tap) / tong                    # |Sv| / |S|
        entropy_v = tinh_entropy(tap, cot_nhan)       # Entropy(Sv)
        entropy_con += trong_so * entropy_v           # Cong don

        # Luu chi tiet
        dem = {}
        for dong in tap:
            n = dong[cot_nhan]
            dem[n] = dem.get(n, 0) + 1
        chi_tiet_con[gia_tri] = {
            'so_mau': len(tap),
            'entropy': entropy_v,
            'trong_so': trong_so,
            'dem_lop': dem
        }

    # Gain = Entropy(goc) - Entropy trung binh cac tap con
    gain = entropy_goc - entropy_con

    return gain, chi_tiet_con, entropy_goc


# ---------------------------------------------------------------------------
# XAY DUNG CAY QUYET DINH (DE QUY)
# ---------------------------------------------------------------------------
def xay_dung_cay(du_lieu, tieu_de, cac_cot_con_lai, cot_nhan, do_sau=0):
    """
    Xay dung cay quyet dinh bang thuat toan ID3 (de quy).
    Tra ve: dict bieu dien cay
      - Nut la: gia tri la nhan lop (vi du: 'yes')
      - Nut trong: dict {'thuoc_tinh': ten, 'nhanh': {gia_tri: cay_con}}
    """
    # --- Dieu kien dung de quy ---

    # Truong hop 1: Tat ca mau cung 1 lop => tra ve nhan do (nut la)
    cac_nhan = set(dong[cot_nhan] for dong in du_lieu)
    if len(cac_nhan) == 1:
        return list(cac_nhan)[0]

    # Truong hop 2: Het thuoc tinh de chia => tra ve lop da so (majority vote)
    if len(cac_cot_con_lai) == 0:
        dem = {}
        for dong in du_lieu:
            n = dong[cot_nhan]
            dem[n] = dem.get(n, 0) + 1
        return max(dem, key=dem.get)

    # Truong hop 3: Khong con du lieu => tra ve None
    if len(du_lieu) == 0:
        return None

    # --- Tim thuoc tinh co Information Gain lon nhat ---
    gain_tot_nhat = -1
    cot_tot_nhat = -1

    for cot in cac_cot_con_lai:
        gain, _, _ = tinh_information_gain(du_lieu, cot, cot_nhan)
        if gain > gain_tot_nhat:
            gain_tot_nhat = gain
            cot_tot_nhat = cot

    # Tao nut cay voi thuoc tinh tot nhat
    ten_thuoc_tinh = tieu_de[cot_tot_nhat]

    # Chia du lieu theo gia tri cua thuoc tinh tot nhat
    tap_con = {}
    for dong in du_lieu:
        gia_tri = dong[cot_tot_nhat]
        if gia_tri not in tap_con:
            tap_con[gia_tri] = []
        tap_con[gia_tri].append(dong)

    # Tao cac nhanh con (de quy)
    cac_cot_moi = [c for c in cac_cot_con_lai if c != cot_tot_nhat]
    nhanh = {}
    for gia_tri, tap in tap_con.items():
        nhanh[gia_tri] = xay_dung_cay(tap, tieu_de, cac_cot_moi, cot_nhan, do_sau + 1)

    return {
        'thuoc_tinh': ten_thuoc_tinh,
        'cot': cot_tot_nhat,
        'nhanh': nhanh
    }


# ---------------------------------------------------------------------------
# IN CAY QUYET DINH (TRUC QUAN HOA TREN TERMINAL)
# ---------------------------------------------------------------------------
def in_cay(cay, tien_to="", la_cuoi=True):
    """
    In cay quyet dinh theo dang cay thu muc (tree view) tren terminal.
    """
    if isinstance(cay, str):
        # Nut la: in nhan lop
        print(f"=> [{cay.upper()}]")
        return

    ten_tt = cay['thuoc_tinh']
    cac_nhanh = list(cay['nhanh'].items())

    for i, (gia_tri, cay_con) in enumerate(cac_nhanh):
        la_nhanh_cuoi = (i == len(cac_nhanh) - 1)
        ky_hieu = "`-- " if la_nhanh_cuoi else "|-- "
        noi = "    " if la_nhanh_cuoi else "|   "

        if isinstance(cay_con, str):
            print(f"{tien_to}{ky_hieu}[{ten_tt} = {gia_tri}] => [{cay_con.upper()}]")
        else:
            print(f"{tien_to}{ky_hieu}[{ten_tt} = {gia_tri}]")
            in_cay(cay_con, tien_to + noi, la_nhanh_cuoi)


# ---------------------------------------------------------------------------
# DU DOAN MAU MOI BANG CAY QUYET DINH
# ---------------------------------------------------------------------------
def du_doan(cay, mau, tieu_de):
    """
    Duyet cay tu goc xuong la de du doan lop cho mau moi.
    Mau la dict: {ten_thuoc_tinh: gia_tri}
    """
    # Nut la => tra ve nhan lop
    if isinstance(cay, str):
        return cay

    ten_tt = cay['thuoc_tinh']
    gia_tri_mau = mau.get(ten_tt, "").lower().strip()

    # Tim nhanh tuong ung voi gia tri cua mau
    if gia_tri_mau in cay['nhanh']:
        return du_doan(cay['nhanh'][gia_tri_mau], mau, tieu_de)
    else:
        # Truong hop gia tri khong ton tai trong cay
        # => tra ve nhan phoi bien nhat (fallback)
        return "khong xac dinh"


# ===========================================================================
# PHAN CHINH: CHAY CHUONG TRINH
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  THUAT TOAN DECISION TREE (ID3) - PHAN LOAI CHOI GOLF")
    print("=" * 70)

    # Doc du lieu
    duong_dan = "golf_df.csv"
    tieu_de, du_lieu = doc_du_lieu(duong_dan)
    cot_nhan = len(tieu_de) - 1  # Cot cuoi cung (Play)

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

    # --- Buoc 1: Tinh Entropy toan bo tap du lieu ---
    entropy_goc = tinh_entropy(du_lieu, cot_nhan)
    dem_lop = {}
    for dong in du_lieu:
        n = dong[cot_nhan]
        dem_lop[n] = dem_lop.get(n, 0) + 1

    print("\n" + "=" * 70)
    print("[BUOC 1] ENTROPY CUA TOAN BO TAP DU LIEU")
    print("-" * 55)
    print(f"  Tong mau: {len(du_lieu)}")
    for lop, sl in dem_lop.items():
        print(f"  Play={lop}: {sl} mau (p = {sl}/{len(du_lieu)} = {sl/len(du_lieu):.4f})")

    cong_thuc = "  Entropy(S) = "
    cac_phan = []
    for lop, sl in dem_lop.items():
        p = sl / len(du_lieu)
        cac_phan.append(f"-({sl}/{len(du_lieu)})*log2({sl}/{len(du_lieu)})")
    cong_thuc += " + ".join(cac_phan)
    print(cong_thuc)
    print(f"  Entropy(S) = {entropy_goc:.4f}")

    # --- Buoc 2: Tinh Information Gain cho tung thuoc tinh ---
    print("\n" + "=" * 70)
    print("[BUOC 2] INFORMATION GAIN CHO TUNG THUOC TINH")
    print("-" * 55)

    cac_cot_thuoc_tinh = [i for i in range(len(tieu_de)) if i != cot_nhan]
    gain_cac_tt = {}

    for cot in cac_cot_thuoc_tinh:
        ten_tt = tieu_de[cot]
        gain, chi_tiet, entropy_s = tinh_information_gain(du_lieu, cot, cot_nhan)
        gain_cac_tt[ten_tt] = gain

        print(f"\n  --- Thuoc tinh: {ten_tt} ---")

        # In chi tiet tung tap con
        entropy_trung_binh_str = []
        for gia_tri, info in chi_tiet.items():
            dem_str = ", ".join(f"{k}={v}" for k, v in info['dem_lop'].items())
            print(f"  {ten_tt}={gia_tri}: {info['so_mau']} mau ({dem_str})")
            print(f"    Entropy({gia_tri}) = {info['entropy']:.4f}")
            entropy_trung_binh_str.append(
                f"({info['so_mau']}/{len(du_lieu)})*{info['entropy']:.4f}"
            )

        print(f"\n  Gain({ten_tt}) = Entropy(S) - [{' + '.join(entropy_trung_binh_str)}]")
        print(f"  Gain({ten_tt}) = {entropy_s:.4f} - {entropy_s - gain:.4f} = {gain:.4f}")

    # --- Buoc 3: Chon thuoc tinh co Gain lon nhat ---
    tt_tot_nhat = max(gain_cac_tt, key=gain_cac_tt.get)
    print("\n" + "=" * 70)
    print("[BUOC 3] SO SANH INFORMATION GAIN")
    print("-" * 40)
    for ten_tt, gain in gain_cac_tt.items():
        dau = " <<< MAX" if ten_tt == tt_tot_nhat else ""
        print(f"  Gain({ten_tt:<14}) = {gain:.4f}{dau}")
    print(f"\n  => Chon thuoc tinh '{tt_tot_nhat}' lam nut goc (Gain lon nhat)")

    # --- Buoc 4: Xay dung cay quyet dinh ---
    print("\n" + "=" * 70)
    print("[BUOC 4] XAY DUNG CAY QUYET DINH (DE QUY)")
    print("-" * 40)

    cay = xay_dung_cay(du_lieu, tieu_de, cac_cot_thuoc_tinh, cot_nhan)

    print(f"\n  Nut goc: [{tt_tot_nhat}]")
    in_cay(cay, tien_to="  ")

    # --- Buoc 5: Du doan mau moi ---
    # Mau can du doan (co the thay doi theo yeu cau giao vien)
    mau_moi = {
        "Outlook": "sunny",
        "Temperature": "cool",
        "Humidity": "high",
        "Windy": "true"
    }

    print("\n" + "=" * 70)
    print("[BUOC 5] DU DOAN MAU MOI")
    print("-" * 40)
    print("  Mau can phan loai:")
    for ten, gia_tri in mau_moi.items():
        print(f"    {ten} = {gia_tri}")

    # Truy vet duong di tren cay
    print("\n  Qua trinh duyet cay:")
    cay_hien_tai = cay
    buoc = 1
    while isinstance(cay_hien_tai, dict):
        ten_tt = cay_hien_tai['thuoc_tinh']
        gia_tri = mau_moi.get(ten_tt, "").lower()
        print(f"    Buoc {buoc}: {ten_tt} = {gia_tri} --> di theo nhanh '{gia_tri}'")
        cay_hien_tai = cay_hien_tai['nhanh'].get(gia_tri, "khong xac dinh")
        buoc += 1

    ket_qua = du_doan(cay, mau_moi, tieu_de)
    print(f"\n  => KET QUA: Play = {ket_qua.upper()}")
    print("=" * 70)
