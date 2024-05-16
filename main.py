import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image_path = 'images/variant-8.jpg'
image = cv2.imread(image_path)

# Проверка загрузки изображения
if image is None:
    raise ValueError(f"Не удалось загрузить изображение по пути: {image_path}")

# Определение центра изображения
height, width, _ = image.shape
center_x, center_y = width // 2, height // 2

# Определение координат для вырезания области 400x400 пикселей
start_x = max(center_x - 200, 0)
start_y = max(center_y - 200, 0)
end_x = min(center_x + 200, width)
end_y = min(center_y + 200, height)

# Вырезание области
cropped_image = image[start_y:end_y, start_x:end_x]

# Сохранение вырезанной области
output_path = 'images/variant-8-cropped.jpg'
cv2.imwrite(output_path, cropped_image)

# Отображение исходного и вырезанного изображений
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Исходное изображение')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Вырезанная область 400x400 пикселей')
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()


import cv2
import numpy as np

def find_marker(image):
    # Преобразование изображения в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Применение гауссового размытия
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Применение порогового значения
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    
    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Выбор самого большого контура
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    return None

def draw_crosshair(image, marker_contour):
    moments = cv2.moments(marker_contour)
    if moments['m00'] == 0:
        return
    
    # Координаты центра метки
    center_x = int(moments['m10'] / moments['m00'])
    center_y = int(moments['m01'] / moments['m00'])
    
    # Отрисовка центра метки
    cv2.circle(image, (center_x, center_y), 10, (0, 255, 0), -1)
    
    # Отрисовка вертикальной и горизонтальной прямых
    cv2.line(image, (center_x, 0), (center_x, image.shape[0]), (255, 0, 0), 2)
    cv2.line(image, (0, center_y), (image.shape[1], center_y), (255, 0, 0), 2)

# Захват видео с камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise ValueError("Не удалось открыть камеру")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    marker_contour = find_marker(frame)
    if marker_contour is not None:
        draw_crosshair(frame, marker_contour)
    
    # Отображение кадра
    cv2.imshow('Tracking', frame)
    
    # Прерывание по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
