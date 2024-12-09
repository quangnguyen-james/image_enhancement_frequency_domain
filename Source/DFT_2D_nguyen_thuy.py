import numpy as np
import matplotlib.pyplot as plt
import cv2

# Định nghĩa hàm biến đổi DFT
def DFT2D(padded):
    M, N = np.shape(padded)
    dft2d = np.zeros((M, N), dtype=complex)
    for k in range(M):
        for l in range(N):
            sum_matrix = 0.0
            for m in range(M):
                for n in range(N):
                    e = np.exp(- 2j * np.pi * (float(k * m) / M + float(l * n) / N))
                    sum_matrix += padded[m, n] * e
            dft2d[k, l] = sum_matrix
    return dft2d

# Định nghĩa hàm biến đổi ngược DFT
def IDFT2D(dft2d):
    M, N = dft2d.shape
    pixels = np.zeros((M, N))
    for m in range(M):
        for n in range(N):
            sum = 0.0
            for k in range(M):
                for l in range(N):
                    e = np.exp(2j * np.pi * (float(k * m) / N + float(l * n) / M))
                    sum += dft2d[l][k] * e

            # get the real part of pixel to show the image
            pixel = sum.real / M / N
            pixels[n, m] = (pixel)
    return pixels

# Định nghĩa hàm lọc thông thấp Ideals
def lowPass_Ideals(D0,U,V):

    # H is our filter
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

    for u in range(U):
        for v in range(V):
            if D[np.abs(u - U0), np.abs(v - V0)] <= D0:
                H[u, v] = 1
            else:
                H[u, v] = 0
    return H

if __name__ == "__main__":

    # Đọc ảnh
    image = cv2.imread("hse1.tiff", 0)
    #image = cv2.resize(src=image, dsize=(70, 70))
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
    dft2d = DFT2D(F_xy_p)
    # Bước 4: Gọi hàm lowPass_Ideals tạo bộ lọc thông thấp Ideals
    H_uv = lowPass_Ideals(5,P,Q)
    # Bước 5: Nhân ảnh sau khi DFT với ảnh sau khi lọc
    G_uv = np.multiply(dft2d, H_uv)
    # Bước 6:
    # Bước 6.1 Thực hiện biến đổi ngược DFT
    g = IDFT2D(G_uv)

    # Bước 6.2: Nhân phần thực ảnh sau khi biến đổi ngược với -1 mũ (x+y)
    g_array = np.asarray(g.real)
    P, Q = np.shape(g_array)
    g_xy_p = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            g_xy_p[x, y] = g_array[x, y] * np.power(-1, x + y)
    # Bước 7: Rút trích ảnh kích thước MxN từ ảnh PxQ
    # Và đây ảnh cuối cùng sau khi lọc
    g_xy = g_xy_p[:shape[0], :shape[1]]

    # Hiển thị ảnh
    fig = plt.figure(figsize=(16, 9))  # Tạo vùng vẽ tỷ lệ 16:9
    #Tạo 9 vùng vẽ con
    (ax1, ax2, ax3), (ax4, ax5, ax6),(ax7, ax8, ax9) = fig.subplots(3, 3)

    # Đọc và hiển thị ảnh gốc
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Ảnh gốc MxN')

    ax2.imshow(f_xy_p, cmap='gray')
    ax2.set_title('Bước 1: Ảnh PxQ')

    ax3.imshow(F_xy_p, cmap='gray')
    ax3.set_title('Bước 2: nhân -1 mũ x+y')

    ax4.imshow(dft2d, cmap='gray')
    ax4.set_title('Bước 3: Phổ tần số ảnh sau khi DFT')

    ax5.imshow(H_uv, cmap='gray')
    ax5.set_title('Bước 4: Phổ tần số Bộ lọc')

    ax6.imshow(G_uv, cmap='gray')
    ax6.set_title('Bước 5: Sau khi nhân DFT với ảnh sau khi lọc ')

    ax7.imshow(g, cmap='gray')
    ax7.set_title('Bước 6.1: Thực hiện biến đổi ngược DFT ')

    ax8.imshow(g_xy_p, cmap='gray')
    ax8.set_title('Bước 6.2: Phần thực sau IDFT nhân -1 mũ (x+y)')

    ax9.imshow(g_xy, cmap='gray')
    ax9.set_title('Bước 7: Ảnh cuối cùng')
    plt.show()