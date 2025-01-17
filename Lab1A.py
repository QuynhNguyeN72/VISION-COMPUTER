import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Hàm so sánh ảnh dựa trên pixel khác biệt
def compare_images(image1, image2):
    # Chuyển cả hai ảnh về ảnh xám nếu cần
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2

    # Resize ảnh nếu kích thước khác nhau
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

    # Tính sự khác biệt giữa hai ảnh
    diff = cv2.absdiff(gray1, gray2)
    _, diff_binary = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

    # Tính số lượng pixel khác biệt
    diff_pixels = np.count_nonzero(diff_binary)
    total_pixels = diff_binary.size

    # Tính phần trăm tương đồng
    similarity_percentage = ((total_pixels - diff_pixels) / total_pixels) 
    return similarity_percentage

# Đọc ảnh
image_path = "C:\Demo\VC\VISION-COMPUTER\distorted_qrcode1.png"
image = cv2.imread(image_path)

reference_image_path = "C:\Demo\VC\VISION-COMPUTER\qr_code_original.png"
reference_image = cv2.imread(reference_image_path)

# Chuyển ảnh sang ảnh xám
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

# Chuyển ảnh sang ảnh nhị phân và tìm đường bao của ảnh
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Tìm đường bao lớn nhất có dạng hình vuông
qr_contour = None
max_area = 0
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True) # Tính chu vi đường bao. True cho biết đường bao là một hình kín.
    approx = cv2.approxPolyDP(contour, epsilon, True) # Tối giản đường bao bằng cách giảm số đỉnh (cạnh) dựa trên khoảng cách epsilon
    
    if len(approx) == 4 and cv2.contourArea(approx) > max_area: # Kiểm tra xem đường bao có đúng 4 đỉnh không?
        qr_contour = approx
        max_area = cv2.contourArea(approx) # Tính diện tích đường bao.

if qr_contour is not None: # Nếu tìm thấy đường bao hợp lệ
    points = qr_contour.reshape(4, 2) # Chuyển các đỉnh đường bao thành mảng tọa độ 2D (4 điểm)
    rect = np.zeros((4, 2), dtype="float32") # Tạo mảng lưu tọa độ các đỉnh sau khi sắp xếp.

    s = points.sum(axis=1) # Tính tổng tọa độ x+y của từng điểm.
    rect[0] = points[np.argmin(s)]  # top-left có tổng nhỏ nhất
    rect[2] = points[np.argmax(s)]  # bottom-right có tổng lớn nhất

    diff = np.diff(points, axis=1) # Tính hiệu tọa độ y−x của từng điểm.
    rect[1] = points[np.argmin(diff)]  # top-right có hiệu nhỏ nhất
    rect[3] = points[np.argmax(diff)]  # bottom-left có hiệu lớn nhất

    width = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
    height = int(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[3] - rect[0])))
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32") # Tạo ma trận đích (tọa độ vuông vắn) cho vùng mã QR sau khi làm phẳng

    matrix = cv2.getPerspectiveTransform(rect, dst) # Tính ma trận biến đổi phối cảnh để "làm phẳng" mã QR.
    warped = cv2.warpPerspective(image, matrix, (width, height)) # Áp dụng phép biến đổi phối cảnh, trả về ảnh mã QR đã được làm phẳn
    
    # So sánh ảnh sau xử lý với ảnh tham chiếu
    similarity_percentage = compare_images(warped, reference_image)
    
    # Đọc nội dung mã QR từ ảnh đã làm phẳng
    decoded_objects = decode(warped)
    qr_content = None
    if decoded_objects:
        for obj in decoded_objects:
            qr_content = obj.data.decode("utf-8")
        print("QR Data:", qr_content)
    else:
        print("Can't decode QR code")

    # In kết quả tương đồng
    print(f"Similarity: {similarity_percentage:.2f}")

    # Hiển thị ảnh
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
    plt.title("Original QR")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("The QR is geometrically transformed")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.title("Warped QR Code")
    plt.axis("off")

    plt.show()
else:
    print("Can't find QR code")
