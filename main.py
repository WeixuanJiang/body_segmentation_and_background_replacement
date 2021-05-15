import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse

def build_argparser():
    parser = argparse.ArgumentParser(description='parameters for body segmentation')
    parser.add_argument('--pt','--confident_threhold',type=float,help='confident threhold for body segmentation model',default=0.5)
    parser.add_argument('--input','--input_path',type=str,help='input path for video or webcam, default is webcam',default='webcam')
    parser.add_argument('-background_img','--background_img_input_path',type=str,help='path for background image',required=True)
    args = parser.parse_args()
    return args


args = build_argparser()

if args.input == 'webcam':
    source = 0
else:
    source = args.input
# download and load the model
bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))
print('Model path: ', BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16)

# read backgroup image
img = cv2.imread(args.background_img)
# read and show webcamera
cap = cv2.VideoCapture(source)
while cap.isOpened():
    ref, frame = cap.read()
    img = img[:frame.shape[0], :frame.shape[1], :]
    if not ref:
        break

    # bodypix detection
    result = bodypix_model.predict_single(frame)
    # create mask
    mask = result.get_mask(threshold=args.pt).numpy().astype(np.int8)
    # apply mask on the frame
    masked_img = cv2.bitwise_and(frame, frame, mask=mask)

    # apply visual backgroud
    neg = np.add(mask, -1)
    inverse = np.where(neg == -1, 1, neg).astype(np.uint8)
    masked_background = cv2.bitwise_and(img, img, mask=inverse)
    finall = cv2.add(masked_img, masked_background)
    cv2.imshow('Bodypix', finall)

    if cv2.waitKey(10) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
