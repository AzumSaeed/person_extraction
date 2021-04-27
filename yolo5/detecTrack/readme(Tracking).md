# People Tracking Engine

----------------------------

This is tracking Engine based on human detection.

Human detection is done by Yolov5 pretrained model, and tracking is done by deep-sort algorithm.

Testing code is as follows.

```python
import cv2
from detecTrack import TrackPerson

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print (TrackPerson.procOneFrame(frame, frame_width=1200))
    cap.release()
```

### Packages and Environments 

###### Python 3.6.5

- opencv-python >= 3.4.x
- PIL
- scipy
- torch >=1.7.1

##### <u>*For real time tracking, Please use GPU*</u>

### To Test this Engine

Execute **__test_tracking_.py**



## References

https://github.com/lewjiayi/Crowd-Analysis

https://github.com/ultralytics/yolov5