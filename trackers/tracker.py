from ultralytics import YOLO
import supervision as sv
import pickle
import os

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
            break
        print(detections[-1])
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
                return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "ball": [],
            "referees": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = self.model.names
            cls_names_inverse = {v:k for k,v in cls_names.items()} 

            #Convert the detections to the format required by the tracker i.e supervision.Detections
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert Goalkeeper to player object
            for object_index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_index] = cls_names_inverse["player"]

            #Track the objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["ball"].append({})
            tracks["referees"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inverse["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                
                if cls_id == cls_names_inverse["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inverse["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

            if stub_path is not None:
                with open(stub_path, "wb") as f:
                    pickle.dump(tracks, f)

            print(tracks)
            return tracks
        
    def draw_ellipse(self , frame, bbox, color, track_id):
        y2 = int(bbox[3])

    def draw_annotations(self, frame, tracks):
        output_video_frams  = []    
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            #Draw the players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 255, 0), track_id)