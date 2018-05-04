import numpy as np
import cv2
from os import path
import pandas as pd
from scipy import stats


def detect_with_corner(image, neighborsN=30, insectsN=6):
    """
    This functions is to find proper insect contours and size from One photo.

    :param image: Single image resource which is used to find objects
    :param neighborsN: To detect corner,
    it apply KNN algorithm. We should find proper neighbors numbers adjusting this parameter. If considering object
    is big, then it needs to increase neighborsN because if we decrease N, then it will provides you several figures
    on one object.s
    :param insectsN: How much insects on one photo, We will sort and extract contours considering this number.
    :return: It returns Pandas Dataframe which consists of area sizes, lists of rectangles points, file path, image shape.
    """

    if not path.isfile(image):
        print("There is no file")

    # image read and rotate it because linux dosen't apply automatic rotation
    img = cv2.imread(image, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    # img = rotate_bound(img, -90)
    img2 = img.copy()
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgray = np.float32(imgray)

    # Corner detection
    dst = cv2.cornerHarris(imgray, neighborsN, 3, 0.04)
    dst = cv2.dilate(dst, None)

    img2[dst > 0.01 * dst.max()] = [0, 0, 255]

    # Just left red with mask
    upper = np.array([0, 0, 255])
    rower = np.array([0, 0, 255])

    mask_red = cv2.inRange(img2, rower, upper)

    # Finding contours: I sorted contours and used to 8th
    _, contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # To remove wrong label on some small spots or big spot, I applied z score statistic
    contours = sorted(contours, key=cv2.contourArea)[-round(insectsN * 1.6):]
    areas10 = list(map(cv2.contourArea, contours))
    rects = list(map(cv2.boundingRect, contours))
    dzip = list(zip(areas10, rects))
    df = pd.DataFrame(dzip, columns=['area', 'rects'])
    df.sort_values(by=['area'])
    df['path'] = image
    df['shape'] = [img.shape]*len(df)

    """
    # if you want to test this function on one picture. then use below codes
    for (x, y, w, h) in rects:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    imageShow(img)
    """

    return df


def imageShow(img):
    """
    To show resized image on middle of the codes.

    :param img: Source image
    :return: None
    """
    img2 = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    cv2.imshow('Auto detected results', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    df = detect_with_corner('/home/paul/Downloads/insects/8_diabrotica_virgifera/20180425_084857.jpg')