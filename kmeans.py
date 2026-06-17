# ===========================================================================
# THUAT TOAN K-MEANS CLUSTERING (PHAN CUM)
# ===========================================================================
# Mo ta:
#   K-Means la thuat toan PHAN CUM (clustering) - HOC KHONG GIAM SAT
#   (unsupervised learning), KHONG phai phan loai (classification).
#   Thuat toan chia du lieu thanh K cum dua tren khoang cach.
#
# LUU Y QUAN TRONG:
#   - K-Means lam viec voi DU LIEU SO (numerical data).
#   - Dataset golf_play la du lieu PHAN LOAI (categorical).
#   - De ap dung K-Means, ta can MA HOA du lieu categorical => so (encoding).
#   - Day la vi du minh hoa, trong thuc te K-Means khong phai lua chon
#     tot nhat cho du lieu hoan toan categorical.
#
# Cong thuc khoang cach Euclid:
#   d(x, y) = sqrt( sum( (xi - yi)^2 ) )
#
# Quy trinh K-Means:
#   Buoc 1: Chon K tam cum ban dau (centroids)
#   Buoc 2: Gan moi diem du lieu vao cum co tam gan nhat
#   Buoc 3: Cap nhat tam cum = trung binh cac diem trong cum
#   Buoc 4: Lap lai Buoc 2-3 cho den khi hoi tu
#            (cac cum khong thay doi hoac dat so vong lap toi da)
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
def tao_bang_ma_hoa_ohe(du_lieu, cac_cot):
    """
    Tao bang ma hoa One-Hot Encoding cho cac cot categorical.
    Tra ve:
      - bang_ma: dict {chi_so_cot: {gia_tri: vector_ohe}}
      - ten_cot_ohe: danh sach ten cac cot sau OHE (de hien thi)
    Vi du: Outlook co 3 gia tri => tao 3 cot: Outlook_overcast, Outlook_rainy, Outlook_sunny
    """
    bang_ma = {}
    ten_cot_ohe = []  # Ten cac cot sau OHE (de in bang)

    for cot in cac_cot:
        # Thu thap tat ca gia tri duy nhat cua cot
        gia_tri_set = []
        for dong in du_lieu:
            if dong[cot] not in gia_tri_set:
                gia_tri_set.append(dong[cot])
        gia_tri_set.sort()  # Sap xep de nhat quan

        so_gia_tri = len(gia_tri_set)

        # Tao vector OHE cho moi gia tri
        # Vi du: 3 gia tri => overcast=[1,0,0], rainy=[0,1,0], sunny=[0,0,1]
        bang_ma[cot] = {}
        for i, gia_tri in enumerate(gia_tri_set):
            vector = [0] * so_gia_tri       # Tao vector toan 0
            vector[i] = 1                    # Dat vi tri thu i = 1
            bang_ma[cot][gia_tri] = vector

        # Luu ten cot OHE
        for gia_tri in gia_tri_set:
            ten_cot_ohe.append(f"{gia_tri}")

    return bang_ma, ten_cot_ohe


def ma_hoa_du_lieu_ohe(du_lieu, bang_ma, cac_cot):
    """
    Chuyen du lieu categorical thanh du lieu so bang One-Hot Encoding.
    Moi dong du lieu duoc chuyen thanh 1 vector nhi phan dai.
    Tra ve: list cac vector so (list of list)
    """
    du_lieu_so = []
    for dong in du_lieu:
        vector = []
        for cot in cac_cot:
            gia_tri = dong[cot]
            # Noi (concatenate) vector OHE cua tung thuoc tinh
            vector.extend(bang_ma[cot][gia_tri])
        du_lieu_so.append(vector)
    return du_lieu_so


# ---------------------------------------------------------------------------
# TINH KHOANG CACH EUCLID
# ---------------------------------------------------------------------------
# Cong thuc: d(x, y) = sqrt( sum( (xi - yi)^2 ) )
# Vi du: x = [0, 1, 0, 1], y = [1, 2, 1, 0]
#   d = sqrt( (0-1)^2 + (1-2)^2 + (0-1)^2 + (1-0)^2 )
#   d = sqrt( 1 + 1 + 1 + 1 ) = sqrt(4) = 2.0
# ---------------------------------------------------------------------------
def khoang_cach_euclid(diem_a, diem_b):
    """
    Tinh khoang cach Euclid giua 2 diem (2 vector so).
    """
    tong = 0
    for i in range(len(diem_a)):
        tong += (diem_a[i] - diem_b[i]) ** 2
    return math.sqrt(tong)


# ---------------------------------------------------------------------------
# KHOI TAO TAM CUM BAN DAU
# ---------------------------------------------------------------------------
# Cach 1: Chon K diem dau tien trong du lieu (don gian, de hieu)
# Cach 2: Chon ngau nhien K diem (thuc te dung cach nay)
# O day dung Cach 1 de ket qua co dinh, de kiem tra khi thi
# ---------------------------------------------------------------------------
def khoi_tao_tam_cum(du_lieu_so, k):
    """
    Chon K diem dau tien lam tam cum ban dau.
    Tra ve: list cac tam cum (moi tam la 1 list so)
    """
    tam_cum = []
    for i in range(k):
        # Sao chep diem (tranh tham chieu)
        tam_cum.append(du_lieu_so[i][:])
    return tam_cum


# ---------------------------------------------------------------------------
# GAN MOI DIEM VAO CUM GAN NHAT
# ---------------------------------------------------------------------------
# Voi moi diem du lieu, tinh khoang cach den tat ca K tam cum,
# roi gan diem vao cum co tam gan nhat.
# ---------------------------------------------------------------------------
def gan_cum(du_lieu_so, tam_cum):
    """
    Gan moi diem du lieu vao cum co tam cum gan nhat.
    Tra ve: list cac chi so cum (0, 1, ..., K-1) tuong ung voi tung diem
    """
    phan_cum = []  # Chi so cum cua tung diem
    for diem in du_lieu_so:
        kc_nho_nhat = float('inf')  # Vo cuc
        cum_gan_nhat = 0

        for j, tam in enumerate(tam_cum):
            kc = khoang_cach_euclid(diem, tam)
            if kc < kc_nho_nhat:
                kc_nho_nhat = kc
                cum_gan_nhat = j

        phan_cum.append(cum_gan_nhat)

    return phan_cum


# ---------------------------------------------------------------------------
# CAP NHAT TAM CUM (TINH TRUNG BINH)
# ---------------------------------------------------------------------------
# Tam cum moi = trung binh cong cac toa do cua tat ca diem trong cum
# Vi du: Cum 1 co cac diem [0,1,0], [2,1,1], [1,0,1]
#   Tam moi = [(0+2+1)/3, (1+1+0)/3, (0+1+1)/3] = [1.0, 0.67, 0.67]
# ---------------------------------------------------------------------------
def cap_nhat_tam_cum(du_lieu_so, phan_cum, k, so_chieu):
    """
    Tinh tam cum moi = trung binh cac diem trong moi cum.
    Tra ve: list tam cum moi
    """
    tam_moi = []
    for j in range(k):
        # Lay tat ca diem thuoc cum j
        cac_diem = [du_lieu_so[i] for i in range(len(du_lieu_so)) if phan_cum[i] == j]

        if len(cac_diem) == 0:
            # Cum rong: giu nguyen tam cu (truong hop hiem gap)
            tam_moi.append([0.0] * so_chieu)
        else:
            # Tinh trung binh tung chieu
            tam = []
            for d in range(so_chieu):
                trung_binh = sum(diem[d] for diem in cac_diem) / len(cac_diem)
                tam.append(round(trung_binh, 4))
            tam_moi.append(tam)

    return tam_moi


# ---------------------------------------------------------------------------
# KIEM TRA HOI TU
# ---------------------------------------------------------------------------
# Hoi tu khi tam cum khong thay doi (hoac thay doi rat nho) giua 2 vong lap
# ---------------------------------------------------------------------------
def da_hoi_tu(tam_cu, tam_moi, nguong=0.0001):
    """
    Kiem tra xem thuat toan da hoi tu chua.
    Hoi tu khi tong khoang cach dich chuyen cua cac tam < nguong.
    """
    tong_dich_chuyen = 0
    for i in range(len(tam_cu)):
        tong_dich_chuyen += khoang_cach_euclid(tam_cu[i], tam_moi[i])
    return tong_dich_chuyen < nguong


# ---------------------------------------------------------------------------
# THUAT TOAN K-MEANS CHINH
# ---------------------------------------------------------------------------
def kmeans(du_lieu_so, k, so_vong_toi_da=20):
    """
    Chay thuat toan K-Means.
    Tra ve: phan_cum (list chi so cum), tam_cum (list cac tam), lich_su (log)
    """
    so_chieu = len(du_lieu_so[0])
    lich_su = []  # Luu lich su moi vong lap

    # Buoc 1: Khoi tao tam cum
    tam_cum = khoi_tao_tam_cum(du_lieu_so, k)

    for vong in range(1, so_vong_toi_da + 1):
        # Buoc 2: Gan moi diem vao cum gan nhat
        phan_cum = gan_cum(du_lieu_so, tam_cum)

        # Buoc 3: Cap nhat tam cum
        tam_moi = cap_nhat_tam_cum(du_lieu_so, phan_cum, k, so_chieu)

        # Luu lich su
        lich_su.append({
            'vong': vong,
            'tam_cum': [t[:] for t in tam_cum],
            'tam_moi': [t[:] for t in tam_moi],
            'phan_cum': phan_cum[:]
        })

        # Buoc 4: Kiem tra hoi tu
        if da_hoi_tu(tam_cum, tam_moi):
            tam_cum = tam_moi
            break

        tam_cum = tam_moi

    return phan_cum, tam_cum, lich_su


# ===========================================================================
# PHAN CHINH: CHAY CHUONG TRINH
# ===========================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  THUAT TOAN K-MEANS CLUSTERING - PHAN CUM DU LIEU GOLF")
    print("=" * 70)

    # Doc du lieu
    duong_dan = "golf_df.csv"
    tieu_de, du_lieu = doc_du_lieu(duong_dan)
    cot_nhan = len(tieu_de) - 1

    # --- In bang du lieu goc ---
    print("\n[BANG DU LIEU GOC]")
    print("-" * 55)
    header_fmt = "{:<6} {:<12} {:<14} {:<10} {:<8} {:<6}"
    print(header_fmt.format("STT", *tieu_de))
    print("-" * 55)
    for i, dong in enumerate(du_lieu, 1):
        print(header_fmt.format(i, *dong))
    print("-" * 55)

    # --- Ma hoa du lieu ---
    # Chi dung cac thuoc tinh (khong dung cot nhan Play)
    cac_cot = [i for i in range(len(tieu_de)) if i != cot_nhan]
    bang_ma, ten_cot_ohe = tao_bang_ma_hoa_ohe(du_lieu, cac_cot)

    print("\n" + "=" * 70)
    print("[BUOC 0] MA HOA DU LIEU CATEGORICAL => SO (One-Hot Encoding)")
    print("-" * 70)
    print("\n  One-Hot Encoding: moi gia tri => 1 vector nhi phan,")
    print("  chi co dung 1 vi tri = 1, con lai = 0.")
    print("  => Khong tao quan he thu tu gia giua cac gia tri.")

    for cot, ma in bang_ma.items():
        ten_cot = tieu_de[cot]
        so_gia_tri = len(ma)
        print(f"\n  {ten_cot} ({so_gia_tri} gia tri => {so_gia_tri} cot OHE):")
        for gia_tri, vector in ma.items():
            print(f"    {gia_tri:<12} => {vector}")

    du_lieu_so = ma_hoa_du_lieu_ohe(du_lieu, bang_ma, cac_cot)
    so_chieu = len(du_lieu_so[0])

    print(f"\n  Du lieu sau khi ma hoa ({so_chieu} chieu):")
    print(f"  Ten cot OHE: {ten_cot_ohe}")
    header2 = "{:<6}" + "{:<4}" * so_chieu
    print("  " + header2.format("STT", *ten_cot_ohe[:so_chieu]))
    print("  " + "-" * (6 + 4 * so_chieu))
    for i, vector in enumerate(du_lieu_so, 1):
        print("  " + header2.format(i, *vector))

    # --- Chay K-Means ---
    K = 2  # So cum (2 cum tuong ung voi yes/no, co the thay doi)
    print("\n" + "=" * 70)
    print(f"[K-MEANS] SO CUM K = {K}")
    print("=" * 70)

    # Buoc 1: Khoi tao tam cum
    print(f"\n[BUOC 1] KHOI TAO TAM CUM BAN DAU")
    print(f"  (Chon {K} diem dau tien lam tam cum)")
    for i in range(K):
        print(f"  Tam cum {i+1} (C{i+1}): {du_lieu_so[i]}")
        # Hien thi du lieu goc tuong ung
        goc = [du_lieu[i][c] for c in cac_cot]
        print(f"    (Tuong ung mau {i+1}: {goc})")

    # Chay thuat toan
    phan_cum, tam_cum_cuoi, lich_su = kmeans(du_lieu_so, K)

    # --- In chi tiet tung vong lap ---
    for log in lich_su:
        vong = log['vong']
        print(f"\n{'=' * 70}")
        print(f"[VONG LAP {vong}]")
        print("-" * 55)

        # In bang khoang cach va phan cum
        print(f"\n  [BUOC 2] TINH KHOANG CACH VA GAN CUM:")
        header3 = "  {:<6}" + "{:<16}" * K + "{:<10}"
        ten_tam = [f"d(x,C{j+1})" for j in range(K)]
        print(header3.format("STT", *ten_tam, "Cum"))
        print("  " + "-" * (6 + 16 * K + 10))

        for i, diem in enumerate(du_lieu_so):
            khoang_cach = []
            for j in range(K):
                kc = khoang_cach_euclid(diem, log['tam_cum'][j])
                khoang_cach.append(f"{kc:.4f}")
            cum = log['phan_cum'][i] + 1  # Hien thi tu 1
            row_vals = [i + 1] + khoang_cach + [f"C{cum}"]
            print(("  {:<6}" + "{:<16}" * K + "{:<10}").format(*row_vals))

        # In cac diem trong moi cum
        print(f"\n  Ket qua phan cum:")
        for j in range(K):
            cac_diem_cum = [i + 1 for i in range(len(du_lieu_so)) if log['phan_cum'][i] == j]
            print(f"    Cum {j+1} (C{j+1}): Mau {cac_diem_cum} ({len(cac_diem_cum)} diem)")

        # In tam cum moi
        print(f"\n  [BUOC 3] CAP NHAT TAM CUM (Trung binh cac diem trong cum):")
        for j in range(K):
            cac_diem = [du_lieu_so[i] for i in range(len(du_lieu_so)) if log['phan_cum'][i] == j]
            print(f"    Tam cum {j+1} cu:  {log['tam_cum'][j]}")
            print(f"    Tam cum {j+1} moi: {log['tam_moi'][j]}")

            # Chi tiet tinh trung binh cho moi chieu
            if len(cac_diem) > 0:
                chi_tiet = []
                for d in range(len(cac_diem[0])):
                    cac_gt = [p[d] for p in cac_diem]
                    tb = sum(cac_gt) / len(cac_gt)
                    chi_tiet.append(f"({'+'.join(str(v) for v in cac_gt)})/{len(cac_gt)}={tb:.4f}")
                print(f"      Tinh: [{', '.join(chi_tiet)}]")
            print()

        # Kiem tra hoi tu
        if da_hoi_tu(log['tam_cum'], log['tam_moi']):
            print(f"  => HOI TU! Tam cum khong thay doi.")
        else:
            dich_chuyen = sum(
                khoang_cach_euclid(log['tam_cum'][j], log['tam_moi'][j])
                for j in range(K)
            )
            print(f"  Tong dich chuyen tam cum: {dich_chuyen:.4f} (chua hoi tu)")

    # --- Ket qua cuoi cung ---
    print("\n" + "=" * 70)
    print("[KET QUA CUOI CUNG]")
    print("-" * 55)
    print(f"  So vong lap: {len(lich_su)}")
    print(f"  So cum K = {K}")
    print()

    # In tam cum cuoi
    for j in range(K):
        print(f"  Tam cum {j+1}: {tam_cum_cuoi[j]}")

    # In ket qua phan cum
    print()
    for j in range(K):
        print(f"  Cum {j+1}:")
        for i in range(len(du_lieu)):
            if phan_cum[i] == j:
                goc = du_lieu[i]
                print(f"    Mau {i+1:>2}: {', '.join(goc[:cot_nhan])} | Play={goc[cot_nhan]}")

    # --- So sanh voi nhan thuc te ---
    print("\n" + "=" * 70)
    print("[SO SANH VOI NHAN THUC TE (Play)]")
    print("-" * 55)
    print("  (K-Means la hoc KHONG GIAM SAT, khong biet nhan thuc te.")
    print("   Phan nay chi de so sanh tham khao.)")
    print()

    for j in range(K):
        dem_nhan = {}
        for i in range(len(du_lieu)):
            if phan_cum[i] == j:
                nhan = du_lieu[i][cot_nhan]
                dem_nhan[nhan] = dem_nhan.get(nhan, 0) + 1

        tong = sum(dem_nhan.values())
        print(f"  Cum {j+1} ({tong} mau):")
        for nhan, sl in dem_nhan.items():
            phan_tram = (sl / tong) * 100
            print(f"    Play={nhan}: {sl} mau ({phan_tram:.1f}%)")

    print("\n" + "=" * 70)
    print("  LUU Y: Du lieu categorical da duoc ma hoa bang One-Hot Encoding.")
    print("  OHE dam bao khoang cach giua cac gia tri khac nhau la nhu nhau,")
    print("  phu hop hon Label Encoding cho K-Means.")
    print("  Tuy nhien, so chieu tang len (4 thuoc tinh => " + str(so_chieu) + " chieu).")
    print("=" * 70)
