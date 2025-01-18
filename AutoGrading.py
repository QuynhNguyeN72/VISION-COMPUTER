
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

column_numbers_10 = ""
column_numbers_11 = ""

# Hàm tìm các hình chữ nhật trong ảnh
def find_rectangles(image, min_area=10000, max_area=100000000):
    edges = cv2.Canny(image, 100, 200)  # Phát hiện cạnh
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min_area < w * h < max_area:  # Lọc diện tích hợp lý
            rectangles.append((x, y, w, h))
    #cv2.imshow("Rectangles", image)
    return rectangles
def display_roi(image, rectangles):
    """
    Hiển thị ảnh với các ROI đã được nhận diện.
    """
    for idx, (x, y, w, h) in enumerate(rectangles):
        # Vẽ hình chữ nhật lên ảnh
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Đánh số thứ tự các ROI
        cv2.putText(image, f"ROI {idx + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    height, width = image.shape[:2]
    new_dimensions = (int(width * 0.22), int(height * 0.2))
    resized_image = cv2.resize(image, new_dimensions)

    # Hiển thị ảnh đã thu nhỏ
    cv2.imshow("Detected ROIs (Scaled)", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# Hàm xử lý khung trắc nghiệm và tạo ma trận với ngưỡng tùy chỉnh
def process_exam_grid_custom(roi, rows=10, cols=6, pixel_threshold=0.3):
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang xám
    roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)  # Làm mịn ảnh
    _, roi_binary = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Nhị phân hóa (đảo ngược)

    height, width = roi_binary.shape
    cell_height = height // rows
    cell_width = width // cols
    matrix = np.zeros((rows, cols), dtype=int)

    for i in range(rows):
        for j in range(cols):
            cell = roi_binary[i * cell_height:(i + 1) * cell_height,
                              j * cell_width:(j + 1) * cell_width]

            # Tính tỷ lệ pixel đen
            black_pixel_ratio = np.sum(cell == 255) / (cell_height * cell_width)

            # Nếu tỷ lệ pixel đen lớn hơn ngưỡng
            if black_pixel_ratio > pixel_threshold:
                matrix[i, j] = 1

    return matrix

def process_exam_grid_custom2(roi, rows=10, cols=6, pixel_threshold=0.3, column_gap=1.2, row_gap=1.1):
    """
    Hàm xử lý ma trận trắc nghiệm với khoảng cách giữa các cột và các hàng lớn hơn một chút
    :param roi: Vùng cần xử lý (ROI)
    :param rows: Số lượng hàng
    :param cols: Số lượng cột
    :param pixel_threshold: Ngưỡng tỷ lệ pixel đen
    :param column_gap: Khoảng cách giãn cách giữa các cột
    :param row_gap: Khoảng cách giãn cách giữa các hàng
    :return: Ma trận kết quả (mảng 2D)
    """
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang xám
    roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)  # Làm mịn ảnh
    _, roi_binary = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Nhị phân hóa (đảo ngược)

    height, width = roi_binary.shape
    cell_height = int(height // (rows * row_gap))  # Điều chỉnh chiều cao mỗi ô với khoảng cách giữa các hàng
    cell_width = int(width // (cols * column_gap))  # Điều chỉnh chiều rộng mỗi ô với khoảng cách giữa các cột
    matrix = np.zeros((rows, cols), dtype=int)

    for i in range(rows):
        for j in range(cols):
            # Lấy từng ô trong ảnh theo vị trí cột và hàng
            cell = roi_binary[i * cell_height:(i + 1) * cell_height,
                              j * cell_width:(j + 1) * cell_width]

            # Tính tỷ lệ pixel đen
            black_pixel_ratio = np.sum(cell == 255) / (cell_height * cell_width)

            # Nếu tỷ lệ pixel đen lớn hơn ngưỡng, đánh dấu là đã tô
            if black_pixel_ratio > pixel_threshold:
                matrix[i, j] = 1

    return matrix

# Hàm chuyển ma trận thành dãy số theo cột cho ROI 10, ROI 11
def matrix_to_column_numbers(matrix, rows, cols):
    result = ""
    for col in range(cols):
        row_index = -1  # Mặc định nếu không tìm thấy hàng nào
        for row in range(rows):
            if matrix[row, col] == 1:
                row_index = row  # Ghi lại hàng có bit 1
                break  # Thoát vòng lặp khi tìm thấy bit 1 đầu tiên
        if row_index == -1:
            result += "0"  # Nếu không có bit 1 nào, gán giá trị 0
        else:
            result += str(row_index)

    return result

# Hàm chuyển ma trận thành dãy số theo cột cho ROI 8, ROI 9 (A, B, C, D)
def matrix_to_column_labels(matrix, rows, cols):
    row_labels = ['A', 'B', 'C', 'D']  # Các lựa chọn
    result = ""

    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 1:
                result += f"{i + 1}{row_labels[j]} "  # Tính toán Hàng Cột, A = 65

    return result.strip()
def matrix_to_column_labels2(matrix, rows, cols):
    row_labels = ['A', 'B', 'C', 'D']  # Các lựa chọn
    result = ""

    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 1:
                result += f"{i + 11}{row_labels[j]} "  # Tính toán Hàng Cột, A = 65

    return result.strip()
def analyze_roi_4(matrix, rows=4, cols=4):
    """
    Phân tích ma trận ROI 4 hoặc ROI 5 sau khi xử lý.
    :param matrix: Ma trận đã cắt còn lại 4x4.
    :param rows: Số hàng (4).
    :param cols: Số cột (4).
    :return: Kết quả phân tích đúng/sai.
    """
    row_labels = ['A', 'B', 'C', 'D']
    results = []

    for col_pair in range(0, cols, 2):  # Xét từng cặp cột
        for row in range(rows):
            if matrix[row, col_pair + 1] == 1:  # Cột phía tay trái (cột 2)
                results.append(f"{col_pair // 2 + 3} {row_labels[row]} : Sai")
            elif matrix[row, col_pair] == 1:  # Cột phía tay phải (cột 1)
                results.append(f"{col_pair // 2 + 3} {row_labels[row]} : Đúng")
    
    return results
def analyze_roi_5(matrix, rows=4, cols=4):
    """
    Phân tích ma trận ROI 4 hoặc ROI 5 sau khi xử lý.
    :param matrix: Ma trận đã cắt còn lại 4x4.
    :param rows: Số hàng (4).
    :param cols: Số cột (4).
    :return: Kết quả phân tích đúng/sai.
    """
    row_labels = ['A', 'B', 'C', 'D']
    results = []

    for col_pair in range(0, cols, 2):  # Xét từng cặp cột
        for row in range(rows):
            if matrix[row, col_pair + 1] == 1:  # Cột phía tay trái (cột 2)
                results.append(f"{col_pair // 2 + 1} {row_labels[row]} : Sai")
            elif matrix[row, col_pair] == 1:  # Cột phía tay phải (cột 1)
                results.append(f"{col_pair // 2 + 1} {row_labels[row]} : Đúng")
    
    return results
def analyze_roi_1(matrix, rows, cols):
    """
    Hàm giải mã ma trận với các bit 1 để chuyển thành chuỗi số hoặc ký tự tương ứng.
    :param matrix: Ma trận 12x30 với các bit 0 hoặc 1.
    :param rows: Số lượng hàng (12).
    :param cols: Số lượng cột (30).
    :return: Chuỗi kết quả giải mã cho mỗi nhóm cột.
    """
    result = ""
    
    for i in range(0, cols, 5):  # Mỗi nhóm có 5 cột
        group = matrix[:, i:i + 5]  # Lấy nhóm 5 cột
        
        # Cắt bỏ cột ngoài cùng bên trái (cột đầu tiên) sau khi lấy nhóm 5 cột
        group = group[:, 1:]  # Giữ lại 4 cột từ cột 2 đến cột 5
        
        # Giải mã cho mỗi nhóm
        group_result = []
        
        # Xử lý từng cột trong nhóm 4 cột
        for col in range(4):
            group_value = ""

            # Kiểm tra hàng 1 (dấu "-")
            if group[0, col] == 1:
                group_value += "-"  # Nếu có bit 1 ở hàng 1, ghi dấu "-"
            
            # Kiểm tra hàng 2 (dấu ",")
            if group[1, col] == 1:
                group_value += ","  # Nếu có bit 1 ở hàng 2, ghi dấu ","

            # Kiểm tra hàng 3 đến 12 (giải mã giá trị 0-9)
            for row in range(2, rows):
                if group[row, col] == 1:
                    group_value += str(row - 2)  # Ghi lại giá trị của hàng từ 0 đến 9
            
            # Nếu không có bit 1 thì thêm "0"
          
            
            # Bỏ số "0" ở đầu nếu không có dấu "-" hoặc ","
           
            
            group_result.append(group_value)

        # In kết quả cho nhóm này
        result += "Câu " + str(i // 5 + 1) + ": " + " ".join(group_result) + "\n"
    
    return result




# Hàm xử lý ROI 10, ROI 11, ROI 9 và ROI 8
def process_answer_sheet(image_path):
    img = cv2.imread(image_path)
   
    if img is None:
        print(f"Lỗi: Không thể tải ảnh từ {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rectangles = find_rectangles(gray)


    # Kiểm tra nếu tồn tại ít nhất 11 ROI
    if len(rectangles) >= 11:  
        # Xử lý ROI 10 (chỉ số 9) với 3 cột
        x10, y10, w10, h10 = rectangles[9]  # ROI thứ 10 (chỉ số bắt đầu từ 0)
        roi10 = img[y10:y10+h10, x10:x10+w10]

        # Gọi hàm xử lý khung trắc nghiệm cho ROI 10 (với ngưỡng pixel 0.3)
        matrix10 = process_exam_grid_custom(roi10, rows=10, cols=3, pixel_threshold=0.3)

        # Chuyển ma trận thành dãy số theo cột cho ROI 10
        column_numbers_10 = matrix_to_column_numbers(matrix10, 10, 3)
        print("Mã Đề: ")
        print(column_numbers_10)
        print(matrix10)
        # Xử lý ROI 11 (chỉ số 10) với 6 cột
        x11, y11, w11, h11 = rectangles[10]  # ROI thứ 11 (chỉ số bắt đầu từ 0)
        roi11 = img[y11:y11+h11, x11:x11+w11]
        cv2.imshow("Binary ROI 10", cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(roi10, cv2.COLOR_BGR2GRAY), (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
        cv2.waitKey(0)
        # Gọi hàm xử lý khung trắc nghiệm cho ROI 11 (với ngưỡng pixel 0.3)
        matrix11 = process_exam_grid_custom(roi11, rows=10, cols=6, pixel_threshold=0.3)
        column_numbers_11 = matrix_to_column_numbers(matrix11, 10, 6)
        print("Số Báo Danh: ")
        print(column_numbers_11)
        print(matrix11)
        # Xử lý ROI 9 (chỉ số 8) với 5 cột và 11 hàng
        x9, y9, w9, h9 = rectangles[8]  # ROI thứ 9 (chỉ số bắt đầu từ 0)
        roi9 = img[y9:y9+h9, x9:x9+w9]

        # Điều chỉnh ma trận cho ROI 9, giảm bớt tỷ lệ phân vùng nếu cần
        matrix9 = process_exam_grid_custom2(roi9, rows=11, cols=5, pixel_threshold=0.12,column_gap=1, row_gap=1)  # Tăng tỷ lệ pixel thấp hơn

        # Cắt cột ngoài cùng tay trái và hàng đầu tiên
        matrix9_cropped = matrix9[1:, 1:]  # Cắt cột và hàng đầu tiên, để còn lại ma trận 10x4
        cv2.imshow("Binary ROI 11", cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(roi11, cv2.COLOR_BGR2GRAY), (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])

        # Hiển thị ma trận nhị phân của ROI 9
        print("Ma trận nhị phân của ROI 9: ")
        print(matrix9_cropped)

        # Chuyển ma trận thành dãy số theo cột cho ROI 9
        column_numbers_9 = matrix_to_column_labels(matrix9_cropped, 10, 4)
        print("Phần 1 ")
        print(column_numbers_9)

        # Hiển thị ảnh nhị phân và các ô để kiểm tra cho ROI 9
        cv2.imshow("Binary ROI 9", cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(roi9, cv2.COLOR_BGR2GRAY), (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
        cv2.waitKey(0)

        # Xử lý ROI 8 (chỉ số 7) với 5 cột và 11 hàng
        x8, y8, w8, h8 = rectangles[7]  # ROI thứ 8 (chỉ số bắt đầu từ 0)
        roi8 = img[y8:y8+h8, x8:x8+w8]

        # Điều chỉnh ma trận cho ROI 8, giảm bớt tỷ lệ phân vùng nếu cần
        matrix8 = process_exam_grid_custom2(roi8, rows=11, cols=5, pixel_threshold=0.12,column_gap=1, row_gap=1)  # Tăng tỷ lệ pixel thấp hơn

        # Cắt cột ngoài cùng tay trái và hàng đầu tiên
        matrix8_cropped = matrix8[1:, 1:]  # Cắt cột và hàng đầu tiên, để còn lại ma trận 10x4

        # Hiển thị ma trận nhị phân của ROI 8
        print("Ma trận nhị phân của ROI 8: ")
        print(matrix8_cropped)

        # Chuyển ma trận thành dãy số theo cột cho ROI 8
        column_numbers_8 = matrix_to_column_labels2(matrix8_cropped, 10, 4)
        #print("Đáp án 11-12: ")
        print(column_numbers_8)
 
        # Hiển thị ảnh nhị phân và các ô để kiểm tra cho ROI 8
        cv2.imshow("Binary ROI 8", cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(roi8, cv2.COLOR_BGR2GRAY), (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
        cv2.waitKey(0)
        print("Phần 2  ")
        x5, y5, w5, h5 = rectangles[4]  # ROI thứ 9 (chỉ số bắt đầu từ 0)
        roi5 = img[y5:y5+h5, x5:x5+w5]

        # Điều chỉnh ma trận cho ROI 9, giảm bớt tỷ lệ phân vùng nếu cần
        matrix5 = process_exam_grid_custom2(roi5, rows=6, cols=5, pixel_threshold=0.12,column_gap=1, row_gap=1)  # Tăng tỷ lệ pixel thấp hơn

        # Cắt cột ngoài cùng tay trái và hàng đầu tiên
        matrix5_cropped = matrix5[2:, 1:]  # Cắt cột và hàng đầu tiên, để còn lại ma trận 10x4
        cv2.imshow("Binary ROI 5", cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(roi5, cv2.COLOR_BGR2GRAY), (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])

        # Hiển thị ma trận nhị phân của ROI 9
        print("Ma trận nhị phân của ROI 5: ")
        print(matrix5_cropped)

        # Chuyển ma trận thành dãy số theo cột cho ROI 9
        column_numbers_5 = analyze_roi_5(matrix5_cropped, rows=4, cols=4)
        print(column_numbers_5)
        x4, y4, w4, h4 = rectangles[3]  # ROI thứ 9 (chỉ số bắt đầu từ 0)
        roi4 = img[y4:y4+h4, x4:x4+w4]

        # Điều chỉnh ma trận cho ROI 9, giảm bớt tỷ lệ phân vùng nếu cần
        matrix4 = process_exam_grid_custom2(roi4, rows=6, cols=5, pixel_threshold=0.13,column_gap=1, row_gap=1)  # Tăng tỷ lệ pixel thấp hơn

        # Cắt cột ngoài cùng tay trái và hàng đầu tiên
        matrix4_cropped = matrix4[2:, 1:]  # Cắt cột và hàng đầu tiên, để còn lại ma trận 10x4
        cv2.imshow("Binary ROI 4", cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(roi4, cv2.COLOR_BGR2GRAY), (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])

        # Hiển thị ma trận nhị phân của ROI 9
        print("Ma trận nhị phân của ROI 4: ")
        print(matrix4_cropped)

        # Chuyển ma trận thành dãy số theo cột cho ROI 9
        column_numbers_4 = analyze_roi_4(matrix4_cropped, rows=4, cols=4)
        print(column_numbers_4)
        print("Phần 3 ")
        x1, y1, w1, h1 = rectangles[0]  # ROI thứ 9 (chỉ số bắt đầu từ 0)
        roi1 = img[y1:y1+h1, x1:x1+w1]

        # Điều chỉnh ma trận cho ROI 9, giảm bớt tỷ lệ phân vùng nếu cần
        matrix1 = process_exam_grid_custom2(roi1, rows=14, cols=30, pixel_threshold=0.1555,column_gap=1, row_gap=1)  # Tăng tỷ lệ pixel thấp hơn

        # Cắt cột ngoài cùng tay trái và hàng đầu tiên
        matrix1_cropped = matrix1[2:, 0:]  # Cắt cột và hàng đầu tiên, để còn lại ma trận 10x4
        cv2.imshow("Binary ROI 1", cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY), (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])

        # Hiển thị ma trận nhị phân của ROI 9
        print("Ma trận nhị phân của ROI 1: ")
        print(matrix1_cropped)
        column_numbers_1 = analyze_roi_1(matrix1_cropped, rows=12, cols=30)
        print(column_numbers_1)
        #result_text.delete('1.0', tk.END)  # Xóa nội dung cũ
        result_text.insert(tk.END, "Mã Đề:\n" + column_numbers_10 + "\n")
        result_text.insert(tk.END, "Số Báo Danh:\n" + column_numbers_11 + "\n")
        result_text.insert(tk.END, "Phần 1:\n" + column_numbers_9 + "\n")
        result_text.insert(tk.END, "" + column_numbers_8 + "\n")
        result_text.insert(tk.END, "Phần 2:\n" + ", ".join(column_numbers_5) + "\n")
        result_text.insert(tk.END, "" + ", ".join(column_numbers_4) + "\n")
        result_text.insert(tk.END, "Phần 3:\n" + column_numbers_1 + "\n")
    else:
        print("Lỗi: Không đủ ROI để xử lý!")

    return column_numbers_10,column_numbers_11


    

def browse_file():
    
    print("Mã Đề:", column_numbers_10)
    print("Số Báo Danh:", column_numbers_11)
    filename = filedialog.askopenfilename(initialdir="/", title="Select a file",
                                          filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
    if filename:
        entry.delete(0, tk.END)
        entry.insert(0, filename)
        process_answer_sheet(filename)  # Gọi hàm xử lý sau khi chọn file

# Tạo cửa sổ giao diện
root = tk.Tk()
root.title("Xử lý bài làm trắc nghiệm")

# Tạo nhãn và ô nhập
label = tk.Label(root, text="Chọn ảnh bài làm:")
label.pack()

entry = tk.Entry(root, width=50)
entry.pack()

# Tạo nút duyệt
button = tk.Button(root, text="Browse", command=browse_file)
button.pack()

# Tạo vùng hiển thị kết quả
result_text = tk.Text(root, height=20, width=50)
result_text.pack()

root.mainloop()
image_path = r"D:\Study\Term1_4th\Thi Giac May Tinh\BaiLamTracNghiem.jpg"
image = cv2.imread(image_path)

# Tìm các ROI
rectangles = find_rectangles(image)

# Hiển thị ảnh với các ROI
display_roi(image.copy(), rectangles)
process_answer_sheet(image_path)
