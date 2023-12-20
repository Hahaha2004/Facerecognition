
import os
import cv2
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
from keras.models import load_model

"""# 

## 1. Đọc thông tin ảnh và chuyển đỏi
"""

def detect_face(img):
    img = img[70:195,78:172]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (50, 50))
    return img

def print_progress(val, val_len, folder, bar_size=20):
    progr = "#"*round((val)*bar_size/val_len) + " "*round((val_len - (val))*bar_size/val_len)
    if val == 0:
        print("", end = "\n")
    else:
        print("[%s] (%d samples)\t label: %s \t\t" % (progr, val+1, folder), end="\r")

dataset_folder = "Q:\BTL_AI_Nhom7\dataset\dataset_folder"

names = []
images = []

# Hàm xử lí tập tin hình ảnh 
for folder in os.listdir(dataset_folder):
    # print(folder)
    folder_path = os.path.join(dataset_folder, folder)
    files = os.listdir(os.path.join(dataset_folder, folder))[:150]
    # print(files)
    if len(files) < 50: # Nếu số lượng ảnh nhỏ hơn 50 , bỏ qua file đó
        continue
    for i, name in enumerate(files):
        if name.find(".jpg") > -1 :
            img_path = os.path.join(folder_path, name)
            img = cv2.imread(img_path)
            img = detect_face(img) 
            if img is not None :
                images.append(img)
                names.append(folder)





"""### 1.1. Biến đổi ảnh:"""

def biendoi_anh(img):
    h, w = img.shape
    center = (w // 2, h // 2)
# Hàm tạo ma trận biến đổi 2D : getRotationMatrix2D
#h: chiều cao ảnh ; w: chiều rộng ảnh
    M_rot_5 = cv2.getRotationMatrix2D(center, 5, 1.0)
    M_rot_neg_5 = cv2.getRotationMatrix2D(center, -5, 1.0)
    M_rot_10 = cv2.getRotationMatrix2D(center, 10, 1.0)
    M_rot_neg_10 = cv2.getRotationMatrix2D(center, -10, 1.0)
#float32: các ma trận biến đổi, để xoay ảnh ở các góc độ khác nhau:
    M_trans_3 = np.float32([[1, 0, 3], [0, 1, 0]])
    M_trans_neg_3 = np.float32([[1, 0, -3], [0, 1, 0]])
    M_trans_6 = np.float32([[1, 0, 6], [0, 1, 0]])
    M_trans_neg_6 = np.float32([[1, 0, -6], [0, 1, 0]])
    M_trans_y3 = np.float32([[1, 0, 0], [0, 1, 3]])
    M_trans_neg_y3 = np.float32([[1, 0, 0], [0, 1, -3]])
    M_trans_y6 = np.float32([[1, 0, 0], [0, 1, 6]])
    M_trans_neg_y6 = np.float32([[1, 0, 0], [0, 1, -6]])

    imgs = []
#Hàm warpAffine: hàm để tạo ảnh mới sau các bước biến đổi ma trận: xoay ảnh,dịch chuyển ảnh, thiết lập giá trị biên mặc định:(255,255)
    imgs.append(cv2.warpAffine(img, M_rot_5, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_rot_neg_5, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_rot_10, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_rot_neg_10, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_3, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_3, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_6, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_6, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_y3, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_y3, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_y6, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.warpAffine(img, M_trans_neg_y6, (w, h), borderValue=(255,255,255)))
    imgs.append(cv2.add(img, 10))
    imgs.append(cv2.add(img, 30))
    imgs.append(cv2.add(img, -10))
    imgs.append(cv2.add(img, -30))
    imgs.append(cv2.add(img, 15))
    imgs.append(cv2.add(img, 45))
    imgs.append(cv2.add(img, -15))
    imgs.append(cv2.add(img, -45))

    return imgs


img_test = images[0]

tangcuong_hinhanh = biendoi_anh(img_test)
# Hàm tăng cường hình ảnh
plt.figure(figsize=(15,10))
for i, img in enumerate(tangcuong_hinhanh):
    plt.subplot(4,5,i+1)
    plt.imshow(img, cmap="gray")
plt.show()

# Hàm tăng cường hình ảnh: Duyệt qua từng hình ảnh trong images và tăng cường chúng bằng hàm biendoi_anh(img) - dòng 65,
#Sau khi tăng cường ảnh, đoạn code sẽ thêm phiên bản tăng cường vào danh sách augmented_images và thêm tên của ảnh gốc ( lặp lại 20)
# lần  vào danh sách augmented_names

augmented_images = []
augmented_names = []
# Xử lí ngoại lệ: 
for i, img in enumerate(images):
    try :
        augmented_images.extend(biendoi_anh(img))
        augmented_names.extend([names[i]] * 20)
    except :
        print(i)

len(augmented_images), len(augmented_names)

images.extend(augmented_images)
names.extend(augmented_names)

len(images), len(names)

unique, counts = np.unique(names, return_counts = True)

for item in zip(unique, counts):
    print(item)

"""## 2. Chuyển đổi nhãn dán    """

le = LabelEncoder()
le.fit(names)
labels = le.classes_
name_vec = le.transform(names)
categorical_name_vec = to_categorical(name_vec)

print("number of class :", len(labels))
print(labels)

print(name_vec)

print(categorical_name_vec)


"""## 3. Tách tập dữ liệu thành 2 phần chính: tập huấn luyện, tập kiểm tra"""

x_train, x_test, y_train, y_test = train_test_split(np.array(images, dtype=np.float32),   # dữ liệu đầu vào
                                                    np.array(categorical_name_vec),       # dữ liệu ra
                                                    test_size=0.3,                          #nâng tỉ lệ test từ 15% lên 30%, vậy tỉ lệ huấn luyện mô hình là 70%
                                                    random_state=42)                        #giông nhau để đảm bảo chia dữ liệu nhất quán->mỗi lần chạy quá trình chia dữ liệu là giống nhau

print(x_train.shape, y_train.shape, x_test.shape,  y_test.shape)                            #in ra kích thước tập test và train

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)          #biến đổi ảnh phù hợp mô hình cnn: thông số :(số mẫu, chiều cao, rộng, kênh==1(vì là ảnh xám))
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_train.shape, x_test.shape

"""## 4. Create Model"""

def cnn_model(input_shape):
    model = Sequential()
   
    model.add(Conv2D(64,            #hàm convolution Layer (conv2d): sử dụng để chọn các đặc trưng của ảnh đầu vào      
                                    #64 bộ lọc, kích thước 3x3, hàm kích hoạt ReLU: max(0,x): đảm bảo các giá trị không tuyến tính đc thông qua, đưa giá trị âm về 0 ở đầu ra của conv2d
                    (3,3),
                    padding="valid",
                    activation="relu",
                    input_shape=input_shape))
    model.add(Conv2D(64,
                    (3,3),
                    padding="valid",
                    activation="relu",
                    input_shape=input_shape))

    model.add(MaxPool2D(pool_size=(2, 2))) #giảm chiều không gian (đưa về ma trận 2x2), giữ lại chi tiết quan trọng, pixel có giá trị lớn nhất

    model.add(Conv2D(128,
                    (3,3),
                    padding="valid",
                    activation="relu"))
    model.add(Conv2D(128,
                    (3,3),
                    padding="valid",
                    activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(len(labels)))  # equal to number of classes
    model.add(Activation("softmax"))   #phân loại đối tượng dựa vào vector đặc trưng trc đó (đưa ảnh từ mảng nhiều chiều về vector)

    model.summary()

    #biên dịch mô hình trước khi bắt đầu huấn luyện
    model.compile(optimizer='adam',       #thuật toán Adam tối ưu mạng nơ-ron
                  loss='categorical_crossentropy',  #hàm mất mát đo lường khoảng cách dự đoán và thực tế
                  metrics = ['accuracy'])    #đánh giá hiệu suất trong quá trình huấn luyện

    return model

"""## 5. Training Model"""

input_shape = x_train[0].shape

EPOCHS = 8
BATCH_SIZE = 32

model = cnn_model(input_shape)

history = model.fit(x_train,
                    y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    validation_split=0.15   # 15% dữ liệu trong bộ dataset đc sd để làm bộ xác thực
                    )

from keras.utils import plot_model
plot_model(model)

def evaluate_model_(history):
    names = [['accuracy', 'val_accuracy'],
             ['loss', 'val_loss']]
    for name in names :
        fig1, ax_acc = plt.subplots()
        plt.plot(history.history[name[0]])
        plt.plot(history.history[name[1]])
        plt.xlabel('Epoch')
        plt.ylabel(name[0])
        plt.title('Model - ' + name[0])
        plt.legend(['Training', 'Validation'], loc='lower right')
        plt.grid()
        plt.show()

evaluate_model_(history)

model.save("model.h5")  #huấn luyện mô hình và đc lưu vào tệp model.h5

y_pred=model.predict(x_test)

#hiển thị biểu đồ đánh giá tỉ lệ
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 8))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# tính toán ma trận sai số
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
np.set_printoptions(precision=2)
'''
# vẽ ma trận sai số chưa chuẩn hóa
plot_confusion_matrix(cnf_matrix, classes=labels,normalize=False,
                      title='Confusion matrix')

print(classification_report(y_test.argmax(axis=1),
                            y_pred.argmax(axis=1),
                            target_names=labels))
'''
"""# Nhận dạng khuôn mặt bằng vdcam"""

def draw_ped(img, label, x0, y0, xt, yt, color=(255,127,0), text_color=(255,255,255)):

    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # vẽ hình chữ nhật lên khuôn mặt
    cv2.rectangle(img,
                  (x0, y0 + baseline),
                  (max(xt, x0 + w), yt),
                  color,
                  2)
    # vẽ tên hình chữ nhật
    cv2.rectangle(img,
                  (x0, y0 - h),
                  (x0 + w, y0 + baseline),
                  color,
                  -1)
    #viết tên
    cv2.putText(img,
                label,
                (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1,
                cv2.LINE_AA)
    return img

# --------- load Haar Cascade model -------------
face_cascade = cv2.CascadeClassifier('Q:\BTL_AI_Nhom7\haarcascades-20231204T142248Z-001\haarcascades\haarcascade_frontalface_default.xml')

# --------- load Keras CNN model -------------
model = load_model("model.h5")
print("[INFO] finish load model...")

#nhận diện khuôn mặt từ video sử dụng webcam

cap = cv2.VideoCapture(0)   #mở webcam
while cap.isOpened() :      #webcam mở thì vòng lặp vẫn chạy
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   #chuyển đổi frame ảnh thành ảnh xám
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)  
        for (x, y, w, h) in faces:

            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (50, 50))
            face_img = face_img.reshape(1, 50, 50, 1)

            result = model.predict(face_img)  #dự đoán class của khuôn mặt. Ví dụ: ông B, anh N
            idx = result.argmax(axis=1)        #lấy chỉ số class có xác suất cao nhất
            confidence = result.max(axis=1)*100     #tính toán độ tin cậy của kq dự đoán
            if confidence > 90:
                label_text = labels[idx]
            else :
                label_text = "N/A"
            label_text = str(label_text)
            frame = draw_ped(frame, label_text, x, y, x + w, y + h, color=(0,255,255), text_color=(50,50,50))

        cv2.imshow('Detect Face', frame)
    else :
        break
    if cv2.waitKey(10) == ord('q'):
        break

cv2.destroyAllWindows()   #đóng cửa sổ đồ họa của OpenCV
cap.release()   #giải phóng tài nguyên webcam

