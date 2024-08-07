import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import NuSVC


def myplot(arr):
    plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
    plt.show()
    plt.close()


def feature_extra(figpath):
    image = cv2.imread(figpath, cv2.IMREAD_GRAYSCALE)
    image = 255 - image

    ret, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    positions = []
    if len(contours) != 0:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # 变量positions：对图像中数字所在位置的简单排序，可能失效
            positions.append(x + y * 3.5)
            image_clip = image[y:y + h, x:x + w]
            edge_h = int(h * 0.25)
            edge_w = (h + edge_h + edge_h - w) * 0.5
            if edge_w < edge_h:
                edge_w = edge_h * 2
            else:
                edge_w = int(edge_w * 0.9)
            constant = cv2.copyMakeBorder(image_clip, edge_h, edge_h, edge_w, edge_w, cv2.BORDER_CONSTANT,
                                          value=(0, 0, 255))
            kernel = np.ones((10, 10))
            imdilate = cv2.dilate(constant, kernel, iterations=1)
            imresize = cv2.resize(imdilate, (28, 28), interpolation=cv2.INTER_AREA)
            digits.append(imresize)

    return digits, positions


def digitsCLF(digits, positions):
    Xdata = np.load("./Dataset/Xdata.npy")
    Ydata = np.load("./Dataset/Ydata.npy")

    clf = NuSVC(nu=0.02, kernel='rbf', gamma=0.02)
    clf.fit(Xdata[0:7000, :], Ydata[0:7000])

    results = []
    for digit in digits:
        scaler = MinMaxScaler(feature_range=(0, 1))
        arr = scaler.fit_transform(digit)

        prediction = clf.predict(arr.reshape(1, -1))
        results.append(int(prediction[0]))

    orders = np.argsort(positions)
    str_digit = ' '
    for i in orders:
        vari = results[i]
        str_digit = str_digit + str(vari)

    print(f'The handwritten digits are:{str_digit}')


def main():
    figpath = "./Figures/Fig (1).png"
    digit, position = feature_extra(figpath)
    if len(position) == 0:
        print('Recognition failed!')
    else:
        digitsCLF(digit, position)


if __name__ == '__main__':
    dt1 = dt.now()
    main()
    print(f'Duration: {dt.now() - dt1}')
