import numpy as np
import cv2
import matplotlib.pyplot as plt

#Phân tích phổ tần số ảnh
def analyze_frequency_spectrum(image_path):
    # Đọc ảnh và chuyển sang grayscale
    img = cv2.imread(image_path, 0)
    
    # Áp dụng FFT2
    f = np.fft.fft2(img)

    # Dịch tần số 0 về trung tâm
    fshift = np.fft.fftshift(f)
    
    # Tính phổ biên độ
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    
    # Tính phổ pha
    phase_spectrum = np.angle(fshift)
    
    # Hiển thị kết quả
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Ảnh gốc')
    # plt.waitforbuttonpress(0)
    
    plt.subplot(132)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Phổ biên độ')
    # plt.waitforbuttonpress(0)

    plt.subplot(133)
    plt.imshow(phase_spectrum, cmap='gray')
    plt.title('Phổ pha')
    # plt.waitforbuttonpress(0)
    
    plt.tight_layout()
    plt.show()
    # Phân tích các thành phần tần số
    def analyze_frequency_components(magnitude_spectrum):
        h, w = magnitude_spectrum.shape
        center_y, center_x = h//2, w//2

        # Tính năng lượng theo vùng tần số
        low_freq = magnitude_spectrum[center_y-h//8:center_y+h//8,
                                    center_x-w//8:center_x+w//8]
        high_freq = magnitude_spectrum[0:h//8, 0:w//8]
        
        return {
            'low_freq_energy': np.mean(low_freq),
            'high_freq_energy': np.mean(high_freq),
            'total_energy': np.mean(magnitude_spectrum)
        }
    
    freq_analysis = analyze_frequency_components(magnitude_spectrum)
    return freq_analysis

# Lớp lọc
class FrequencyDomainFilter:
    def __init__(self, shape):
        self.shape = shape
        self.center = (shape[0]//2, shape[1]//2)
        self.D = self._create_distance_matrix()
    
    def _create_distance_matrix(self):

        rows, cols = self.shape
        # Tạo mảng tọa độ u, v tương đối so với tâm
        u = np.arange(rows) - self.center[0]
        v = np.arange(cols) - self.center[1]

        # Tạo lưới tọa độ 2D
        u, v = np.ogrid[-rows//2:rows//2, -cols//2:cols//2]
        # u, v = np.meshgrid(u, v)
        
        # Tính ma trận khoảng cách
        return np.sqrt(u**2 + v**2)
    
    def ideal_lowpass(self, D0):    #D0 Là tần số cắt (cutoff frequency)
        """Bộ lọc thông thấp lý tưởng"""
        H = np.zeros(self.shape, dtype=np.float32)
        mask = self.D <= D0
        H[mask] = 1
        return H
    
    def butterworth_lowpass(self, D0, n=2):
        """Bộ lọc thông thấp Butterworth"""
        return 1 / (1 + (self.D/D0)**(2*n))
    
    def gaussian_lowpass(self, D0):
        """Bộ lọc thông thấp Gaussian"""
        return np.exp(-(self.D**2)/(2*D0**2))
    
    def ideal_highpass(self, D0):
        """Bộ lọc thông cao lý tưởng"""
        return 1 - self.ideal_lowpass(D0)
    
    def butterworth_highpass(self, D0, n=2):
        """Bộ lọc thông cao Butterworth"""
        return 1 - self.butterworth_lowpass(D0, n)
    
    def gaussian_highpass(self, D0):
        """Bộ lọc thông cao Gaussian"""
        return 1 - self.gaussian_lowpass(D0)

    def bandpass(self, D0_low, D0_high):
        """Bộ lọc thông dải"""
        return self.ideal_highpass(D0_low) * self.ideal_lowpass(D0_high)
    
def apply_filter(image, filter_func):
    """Áp dụng bộ lọc trong miền tần số"""
    # Chuyển sang miền tần số
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    # Áp dụng bộ lọc
    filtered = fshift * filter_func
    
    # Chuyển ngược về miền không gian
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

def demonstrate_filters(image):
    """Minh họa các bộ lọc khác nhau"""
    # Khởi tạo bộ lọc
    filter_obj = FrequencyDomainFilter(image.shape)
    D0 = 30  # Tần số cắt
    
    # Tạo các bộ lọc
    filters = {
        'Ideal Lowpass': filter_obj.ideal_lowpass(D0),
        'Butterworth Lowpass': filter_obj.butterworth_lowpass(D0),
        'Gaussian Lowpass': filter_obj.gaussian_lowpass(D0),
        'Ideal Highpass': filter_obj.ideal_highpass(D0),
        'Butterworth Highpass': filter_obj.butterworth_highpass(D0),
        'Gaussian Highpass': filter_obj.gaussian_highpass(D0),
        'Bandpass': filter_obj.bandpass(D0/2, D0*2)
    }
    
    # Áp dụng và hiển thị kết quả
    results = {}
    for name, filter_func in filters.items():
        filtered_img = apply_filter(image, filter_func)
        results[name] = filtered_img
    
    return results

def show_filter_responses():
    """Hiển thị đáp ứng tần số của các bộ lọc"""
    x = np.linspace(0, 100, 1000)
    D0 = 50
    
    # Tính đáp ứng
    ideal = np.where(x <= D0, 1, 0)
    butterworth = 1 / (1 + (x/D0)**(2*2))
    gaussian = np.exp(-(x**2)/(2*D0**2))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, ideal, 'b-', label='Ideal')
    plt.plot(x, butterworth, 'r-', label='Butterworth')
    plt.plot(x, gaussian, 'g-', label='Gaussian')
    plt.xlabel('Frequency')
    plt.ylabel('Response')
    plt.title('Frequency Response of Different Lowpass Filters')
    plt.legend()
    plt.grid(True)
    plt.show()

# Ví dụ sử dụng nâng cao chất lượng ảnh
def enhance_image(image):
    """Nâng cao chất lượng ảnh bằng kết hợp các bộ lọc"""
    filter_obj = FrequencyDomainFilter(image.shape)
    
    # Tạo bộ lọc kết hợp
    D0_low = 50
    D0_high = 10
    alpha = 1.5  # Hệ số tăng cường
    
    # Lọc thông thấp để giảm nhiễu
    H_low = filter_obj.gaussian_lowpass(D0_low)
    
    # Tăng cường biên
    H_high = filter_obj.gaussian_highpass(D0_high)
    
    # Kết hợp các bộ lọc
    H_combined = H_low + alpha * H_high
    
    # Áp dụng bộ lọc
    return apply_filter(image, H_combined)

def show_demo_filters(image_path):
    # Đọc ảnh và chuyển sang grayscale
    # image_path = 'D:\Learn\ThS\TaiLieu\HK1.24\CV\Code\HInhThe.jpg'
    img = cv2.imread(image_path, 0)

    plt.figure(figsize=(15,5))
    i=2

    plt.subplot(2,4,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.colorbar()

    demo_filters = demonstrate_filters(img)
    for filter_name,filtered_img in demo_filters.items():
        print(i)
        plt.subplot(2,4,i)
        plt.imshow(filtered_img, cmap='gray')
        plt.title(filter_name)
        plt.colorbar()
        i = i+1
        
    plt.tight_layout()
    plt.show()

def show_enhance_image(image_path):
    # Đọc ảnh và chuyển sang grayscale
    img = cv2.imread(image_path, 0)

    plt.figure(figsize=(15,5))

    enhanced_image = enhance_image(img)

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title('enhanced_image')
    plt.colorbar()
        
    plt.tight_layout()
    plt.show()

# Demo hiển thị phổ
#analyze_frequency_spectrum('ConBao.jpg')

# Demo hiển thị các bộ lọc
show_demo_filters('EmBe.jpeg')

