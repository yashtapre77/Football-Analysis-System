from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read the video
    frames = read_video('inp_vid/train1.mp4')
    print("Frames read successfully")

    #Create an instance of the tracker
    tracker = Tracker("models/best.pt")

    print("Tracker created successfully")
    # detections = tracker.detect_frames(frames)
    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl")
    print("Detections done successfully")
    # print(tracks)

    # Draw the annotations
    output_frames = tracker.draw_annotations(frames, tracks)
    # Save the video
    save_video(output_frames, 'out_vid/train1.avi')

if __name__ == '__main__':
    main()