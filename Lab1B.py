import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode
import sys
sys.stdout.reconfigure(encoding='utf-8')

def compare_images(image1, image2):
    # Chuyển cả hai ảnh về cùng kích thước và định dạng xám
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    # So sánh hai ảnh
    diff = cv2.absdiff(gray1, gray2)
    _, diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Tính phần trăm tương đồng
    diff_pixels = np.count_nonzero(diff)
    total_pixels = diff.size
    similarity_percentage = ((total_pixels - diff_pixels) / total_pixels)
    return similarity_percentage

# Đọc ảnh
image_path = "C:\Demo\VC\VISION-COMPUTER\distorted_qrcode2.png"
image = cv2.imread(image_path)

reference_image_path = "C:\Demo\VC\VISION-COMPUTER\qr_code_original.png"
reference_image = cv2.imread(reference_image_path)

# Lọc nhiễu ảnh
blurred = cv2.GaussianBlur(image, (7, 7), 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))

# Phân ngưỡng bằng Otsu
_, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Khôi phục cấu trúc QR
kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# So sánh với ảnh tham chiếu
similarity_percentage = compare_images(reference_image, eroded)

 # Đọc nội dung mã QR từ ảnh đã làm phẳng
decoded_objects = decode(eroded)
qr_content = None
if decoded_objects:
    for obj in decoded_objects:
        qr_content = obj.data.decode("utf-8")
    print("QR Data:", qr_content)
else:
    print("Can't decode QR code")
        
# Hiển thị kết quả
print(f"Similarity: {similarity_percentage:.2f}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
plt.title("Original QR")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("QR with noise")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(eroded, cmap='gray')
plt.title("Filtered QR")
plt.axis('off')

plt.show()
