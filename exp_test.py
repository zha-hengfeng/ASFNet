


import  cv2

image_ori = cv2.imread("result/cityscapes/apfnetv2r34c32backbone/frankfurt_000000_000294_leftImg8bit.png", cv2.IMREAD_GRAYSCALE)
image_color = cv2.applyColorMap(image_ori, cv2.COLORMAP_JET)
cv2.imshow("img", image_color)
cv2.waitKey()