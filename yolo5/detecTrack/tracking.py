import numpy as np
import cv2

from detecTrack.deep_sort import nn_matching
from detecTrack.deep_sort.detection import Detection
from detecTrack.deep_sort.tracker import Tracker
from detecTrack.deep_sort import generate_detections as gdet
from detecTrack.psdetect.body_detect import detectPerson
from detecTrack.yolo5.detect import detect

def detect_human (frame, encoder, tracker, time, confthreshold = 0.6, nmsthreshold=0.3):
	# Get the dimension of the frame
	(frame_height, frame_width) = frame.shape[:2]
	# Initialize lists needed for detection
	boxes, confidences, centroids = detect(frame, confthread=confthreshold)
	# Perform Non-maxima suppression to suppress weak and overlapping boxes
	# It will filter out unnecessary boxes, i.e. box within box
	# Output will be indexs of useful boxes(xywh)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthreshold, nmsthreshold)

	tracked_bboxes = []
	expired = []
	if len(idxs) > 0:
		del_idxs = []
		for i in range(len(boxes)):
			if i not in idxs:
				del_idxs.append(i)
		for i in sorted(del_idxs, reverse=True):
			del boxes[i]
			del centroids[i]
			del confidences[i]

		boxes = np.array(boxes)
		centroids = np.array(centroids)
		confidences = np.array(confidences)
		features = np.array(encoder(frame, boxes))
		detections = [Detection(bbox, score, centroid, feature) for bbox, score, centroid, feature in zip(boxes, confidences, centroids, features)]

		tracker.predict()
		expired = tracker.update(detections, time)

		# Obtain info from the tracks
		for track in tracker.tracks:
				if not track.is_confirmed() or track.time_since_update > 5:
						continue 
				tracked_bboxes.append(track)

	return [tracked_bboxes, expired]

