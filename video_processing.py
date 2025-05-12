# video_processor.py
import cv2
import numpy as np
from ultralytics import YOLO

PARKING_AREAS = {
    1: [(52,364),(30,417),(73,412),(88,369)],
    2: [(105,353),(86,428),(137,427),(146,358)],
    3: [(159,354),(150,427),(204,425),(203,353)],
    4: [(217,352),(219,422),(273,418),(261,347)],
    5: [(274,345),(286,417),(338,415),(321,345)],
    6: [(336,343),(357,410),(409,408),(382,340)],
    7: [(396,338),(426,404),(479,399),(439,334)],
    8: [(458,333),(494,397),(543,390),(495,330)],
    9: [(511,327),(557,388),(603,383),(549,324)],
    10: [(564,323),(615,381),(654,372),(596,315)],
    11: [(616,316),(666,369),(703,363),(642,312)],
    12: [(674,311),(730,360),(764,355),(707,308)]
}



def load_model():
    return YOLO('yolov8s.pt')

def process_frame(frame, model):
    # (Same logic as before, but structured for Streamlit)
    space_status = {i: False for i in range(1, 13)}
    results = model.predict(frame, verbose=False)
    boxes = results[0].boxes.data.cpu().numpy()
    
    for box in boxes:
        x1, y1, x2, y2, _, class_id = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        for area_id, area_points in PARKING_AREAS.items():
            if cv2.pointPolygonTest(np.array(area_points, np.int32), (cx, cy), False) >= 0:
                space_status[area_id] = True
    return space_status

def draw_parking_overlay(frame, space_status):
    # (Same drawing logic as before)
    return frame