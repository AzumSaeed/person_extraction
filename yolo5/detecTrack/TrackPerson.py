
import datetime
import time
import numpy as np
import imutils
import cv2
import os
import csv
import json
from detecTrack.deep_sort import nn_matching
from detecTrack.deep_sort.tracker import Tracker
from detecTrack.deep_sort import generate_detections as gdet
from detecTrack.tracking import detect_human
from detecTrack import colors

# frame size 480-1920
#initialize deep sort object
TRACK_MAX_AGE = 3
# Video Path
VIDEO_CONFIG = {
	"IS_CAM" : False,
	"CAM_APPROX_FPS": 3
}
max_age = VIDEO_CONFIG["CAM_APPROX_FPS"] * TRACK_MAX_AGE
if max_age > 30:
	max_age = 30

dir_path = os.path.dirname(os.path.realpath(__file__))
model_filename = dir_path + '/mars-small128.pb'
nn_budget = None
max_cosine_distance=0.7
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric, max_age=max_age)


def procOneFrame(imgframe, frame_width = 0, bShowDetect = True, bShowTrackId = True, bShowResult = True, showResultCallback = None):

	if frame_width > 0:
		frame = imutils.resize(imgframe, width=frame_width)
	else:
		frame = imgframe.copy()
	# Get current time
	current_datetime = datetime.datetime.now()
	# Run tracking algorithm
	[humans_detected, expired] = detect_human(frame, encoder, tracker, current_datetime)

	# Record movement data
	rectrjResult = []
	for movement in expired:
		track_id = movement.track_id
		entry_time = movement.entry
		exit_time = movement.exit
		positions = movement.positions
		positions = np.array(positions).flatten()
		positions = list(positions)
		rectrjResult.append([track_id, [entry_time, exit_time], positions])

	trackResult = []
	# Initiate video process loop
	for i, track in enumerate(humans_detected):
		# Get object bounding box
		[x, y, w, h] = list(map(int, track.to_tlbr().tolist()))
		# Get object centroid
		[cx, cy] = list(map(int, track.positions[-1]))
		# Get object id
		idx = track.track_id
		# Draw yellow boxes for detection with social distance violation, green boxes for no violation
		# Place a number of violation count on top of the box
		if bShowDetect:
			cv2.rectangle(frame, (x, y), (w, h), colors.RGB_COLORS["green"], 2)
		if bShowTrackId:
			cv2.putText(frame, str(int(idx)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
						0.8, colors.RGB_COLORS["green"], 2)
		# save result
		trackResult.append([idx, [cx, cy], [x, y, w, h]])

	# Display crowd count on screen
	if bShowResult or showResultCallback:
		text = "Crowd count: {}".format(len(humans_detected))
		cv2.putText(frame, text, (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
		if bShowResult:
			cv2.imshow("Result", frame)
		if showResultCallback:
			showResultCallback(frame)
		cv2.waitKey(1)

	# Display current time on screen
	# current_date = str(current_datetime.strftime("%b-%d-%Y"))
	# current_time = str(current_datetime.strftime("%I:%M:%S %p"))
	# cv2.putText(frame, (current_date), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
	# cv2.putText(frame, (current_time), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
	return trackResult, rectrjResult



