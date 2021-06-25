import numpy as np
import cv2



#inWidth = 500
#inHeight = 500
#WHRatio = inWidth / float(inHeight)
#inScaleFactor = 0.008743
#meanVal = 127.5



thr = 0.45

weights = 'frozen_inference_graph.pb'
prototxt = 'ssd_mobilenet_v1_coco.pbtxt'

classNames = {0: 'background',
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
            7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
            13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
            18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
            24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
            32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
            37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
            41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
            46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
            51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
            56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
            61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
            67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
            75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
            80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
            86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

net = cv2.dnn.readNetFromTensorflow(weights, prototxt)
swapRB = False
cap = cv2.VideoCapture(0)

while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        obj_list = []
        cols = frame.shape[1]
        rows = frame.shape[0]

        #if cols / float(rows) > WHRatio:
        #    cropSize = (int(rows * WHRatio), rows)
        #else:
        #    cropSize = (cols, int(cols / WHRatio))

        #y1 = int((rows - cropSize[1]) / 2)
        #y2 = y1 + cropSize[1]
        #x1 = int((cols - cropSize[0]) / 2)
        #x2 = x1 + cropSize[0]
        #frame = frame[y1:y2, x1:x2]

        cols = frame.shape[1]
        rows = frame.shape[0]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > thr:
                class_id = int(detections[0, 0, i, 1])

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop   = int(detections[0, 0, i, 5] * cols)
                yRightTop   = int(detections[0, 0, i, 6] * rows)

                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0))
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                         (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                         (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    obj_list.append(classNames[class_id])
                                
       
        for j, (pred_class, n_class) in enumerate(x for x in set(map(lambda x: (x, obj_list.count(x)), obj_list))):
            cv2.putText(img=frame,
                        text='{}: {}'.format(pred_class, n_class),
                        org=(5, 40 * (j + 1)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0,255,0),
                        thickness=2)
                                
         
        cv2.imshow("detections", frame)
        if cv2.waitKey(1) >= 0:
            break
