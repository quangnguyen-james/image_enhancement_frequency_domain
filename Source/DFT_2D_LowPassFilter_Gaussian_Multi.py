import numpy as np
import matplotlib.pyplot as plt
import cv2

# Định nghĩa hàm biến đổi DFT
def DFT1D(img):
    U = len(img)
    outarry = np.zeros(U, dtype=complex)
    for m in range(U):
        sum = 0.0
        for n in range(U):
            e = np.exp(-1j * 2 * np.pi * m * n / U)
            sum += img[n] * e
        outarry[m] = sum
    return outarry

def IDFT1D(img):
    U = len(img)
    outarry = np.zeros(U,dtype=complex)
    for n in range(U):
        sum = 0.0
        for m in range(U):
            e = np.exp(1j * 2 * np.pi * m * n / U)
            sum += img[m]*e
        pixel = sum/U
        outarry[n]=pixel
    return outarry

# Định nghĩa hàm lọc thông thấp GaussianLP
def GaussianLP(D0,U,V):
    # H cho filter
    H = np.zeros((U, V))
    D = np.zeros((U, V))
    U0 = int(U / 2)
    V0 = int(V / 2)
    # Tính khoảng cách
    for u in range(U):
        for v in range(V):
            u2 = np.power(u, 2)
            v2 = np.power(v, 2)
            D[u, v] = np.sqrt(u2 + v2)
    # Tính bộ lọc
    for u in range(U):
        for v in range(V):
            H[u, v] = np.exp((-D[np.abs(u - U0), np.abs(v - V0)]**2)/(2*(D0**2)))

    return H
#Định nghĩa hàm chuyển đổi Gaussian
def GaussianLowPass(dft,H,P1,Q1):
     # Bước 5: Nhân ảnh sau khi DFT với ảnh sau khi lọc
    G_uv1 = np.multiply(dft, H)
     # Bước 6:
    # Bước 6.1 Thực hiện biến đổi ngược DFT
    idft_cot = idft_hang = np.zeros((P1, Q1))
    # chuyển đổi DFT ngược theo P - theo cột
    for i in range(P1):
        idft_cot[i] = IDFT1D(G_uv1[i])
    # Chuyển đổi DFT ngược theo Q - theo hàng
    for j in range(Q1):
        idft_hang[:, j] = IDFT1D(idft_cot[:, j])

    # Bước 6.2: Nhân phần thực ảnh sau khi biến đổi ngược với -1 mũ (x+y)
    g_array = np.asarray(idft_hang.real)
    P1, Q1 = np.shape(g_array)
    g_xy_p = np.zeros((P1, Q1))
    for x in range(P1):
        for y in range(Q1):
            g_xy_p[x, y] = g_array[x, y] * np.power(-1, x + y)

    # Bước 7: Rút trích ảnh kích thước MxN từ ảnh PxQ
    # Và đây ảnh cuối cùng sau khi lọc
    g_xy = g_xy_p[:shape[0], :shape[1]]
    return g_xy

if __name__ == "__main__":
    # Đọc ảnh
    image = cv2.imread("QuangTest.jpg", 0)
    image = cv2.resize(src=image, dsize=(100, 100))
    # Chuyển các pixel của ảnh vào mảng 2 chiều f
    f = np.asarray(image)
    M, N = np.shape(f)  # Chiều x và y của ảnh

    # Bước 1: Chuyển ảnh từ kích thước MxN vào ảnh PxQ với P= 2M và Q =2N
    P, Q = 2*M , 2*N
    shape = np.shape(f)
    # Chuyển ảnh PxQ vào mảng fp
    f_xy_p = np.zeros((P, Q))
    f_xy_p[:shape[0], :shape[1]] = f

    # Bước 2: Nhân ảnh fp(x,y) với (-1) mũ (x+y) để tạo ảnh mới
    # Kết quả nhân lưu vào ma trận ảnh fpc
    F_xy_p = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            F_xy_p[x, y] = f_xy_p[x, y] * np.power(-1, x + y)

    # Bước 3: Chuyển đổi ảnh Fpc sang miền tần số (DFT)
    dft_cot = dft_hang = np.zeros((P, Q))
    # DFT theo P - theo cột
    for i in range(P):
        dft_cot[i] = DFT1D(F_xy_p[i])
    # DFT theo Q - theo hàng
    for j in range(Q):
        dft_hang[:, j] = DFT1D(dft_cot[:, j])

    # Bước 4: Gọi hàm GaussianLP tạo bộ lọc thông thấp Gaussian
    H_uv30 = GaussianLP(30,P,Q)    
    G_xy30 = GaussianLowPass(dft_hang,H_uv30,P,Q)
    H_uv60 = GaussianLP(60,P,Q)    
    G_xy60 = GaussianLowPass(dft_hang,H_uv60,P,Q)
    H_uv90 = GaussianLP(90,P,Q)    
    G_xy90 = GaussianLowPass(dft_hang,H_uv90,P,Q)
    H_uv120 = GaussianLP(120,P,Q)    
    G_xy120 = GaussianLowPass(dft_hang,H_uv120,P,Q)
    H_uv150 = GaussianLP(150,P,Q)    
    G_xy150 = GaussianLowPass(dft_hang,H_uv150,P,Q)
    H_uv180 = GaussianLP(180,P,Q)    
    G_xy180 = GaussianLowPass(dft_hang,H_uv180,P,Q)
    H_uv210 = GaussianLP(210,P,Q)    
    G_xy210 = GaussianLowPass(dft_hang,H_uv210,P,Q)
    H_uv240 = GaussianLP(240,P,Q)    
    G_xy240 = GaussianLowPass(dft_hang,H_uv240,P,Q)

    # Hiển thị ảnh
    fig = plt.figure(figsize=(16, 9))  # Tạo vùng vẽ tỷ lệ 16:9
    # Tạo 9 vùng vẽ con
    (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = fig.subplots(3, 3)

    # Đọc và hiển thị ảnh gốc
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image MxN')
    ax1.axis('off')
    # Hiển thị ảnh sau khi Gaussian Low Pass với D0 = 30
    ax2.imshow(G_xy30, cmap='gray')
    ax2.set_title('D0 = 30')
    ax2.axis('off')
    # Hiển thị ảnh sau khi Gaussian Low Pass với D0 = 60
    ax3.imshow(G_xy60, cmap='gray')
    ax3.set_title('D0 = 60')
    ax3.axis('off')
    # Hiển thị ảnh sau khi Gaussian Low Pass với D0 = 90
    ax4.imshow(G_xy90, cmap='gray')
    ax4.set_title('D0 = 90')
    ax4.axis('off')
    # Hiển thị ảnh sau khi Gaussian Low Pass với D0 = 120
    ax5.imshow(G_xy120, cmap='gray')
    ax5.set_title('D0 = 120')
    ax5.axis('off')
    # Hiển thị ảnh sau khi Gaussian Low Pass với D0 = 150
    ax6.imshow(G_xy150, cmap='gray')
    ax6.set_title('D0 = 150')
    ax6.axis('off')
    # Hiển thị ảnh sau khi Gaussian Low Pass với D0 = 180
    ax7.imshow(G_xy180, cmap='gray')
    ax7.set_title('D0 = 180')
    ax7.axis('off')
    # Hiển thị ảnh sau khi Gaussian Low Pass với D0 = 210
    ax8.imshow(G_xy210, cmap='gray')
    ax8.set_title('D0 = 210')
    ax8.axis('off')
    # Hiển thị ảnh sau khi Gaussian Low Pass với D0 = 240
    ax9.imshow(G_xy240, cmap='gray')
    ax9.set_title('D0 = 240')
    ax9.axis('off')

    plt.show()