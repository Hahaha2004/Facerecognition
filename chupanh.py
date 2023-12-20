import os
import cv2

dataset_folder = "Q:\\BTL_AI_Nhom7\\dataset\\dataset_folder"

cap = cv2.VideoCapture(0)

my_name = "DaoDuck"
os.makedirs(os.path.join(dataset_folder, my_name), exist_ok=True)
soluonganhchup = 90

i = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret :
        cv2.imshow("Capture Photo", frame)
        cv2.imwrite(f"{dataset_folder}\\{my_name}\\{my_name}_{i:04d}.jpg", cv2.resize(frame, (250, 250)))

        if cv2.waitKey(1000) == ord('q') or i == soluonganhchup:
            break
        i += 1
cap.release()
cv2.destroyAllWindows()