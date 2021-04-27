import numpy as np
import tensorflow as tf
import cv2
import time
import os

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time For Detection:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


# main flow
dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = dir_path + '/frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)

def detectPerson(imgframe, confthreshold = 0.7):
    imgcpy = imgframe.copy()
    boxes, scores, classes, num = odapi.processFrame(imgcpy)
    final_score = np.squeeze(scores)
    count = 0
    bboxes, bscores = [], []
    # Rect of detection
    for i in range(len(boxes)):
        if classes[i] == 1 and scores[i] > confthreshold:
            bboxes.append(boxes[i])
            bscores.append(scores[i])
    # bboxes(ltrb)
    return bboxes, bscores

def releaseModel():
    odapi.close()

if __name__ == "__main__":

    cap = cv2.VideoCapture('E:\\_WORK_Upwork_Evgenii\\Asad Riaz\\002/(4).mp4')
    while cap.isOpened():
        r, img = cap.read()
        # img = cv2.resize(img, (480, 848))
        img = cv2.resize(img, (1200, 800))
        # img = ndimage.rotate(img, 90)
        bboxes, bscores = detectPerson(img)
        for ii in range(len(bboxes)):
            bx = bboxes[ii]
            bs = bscores[ii]
            cv2.rectangle(img, (bx[1], bx[0]), (bx[3], bx[2]),
                          (0, 0, 255), 5)
            croped = img[bx[0]:bx[2], bx[1]:bx[3]]
            cv2.imshow(str(ii), croped)

        cv2.imshow("main", img)
        cv2.waitKey(1)