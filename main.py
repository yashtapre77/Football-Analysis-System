from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
def main():
    # Read the video
    frames = read_video('inp_vid/train.mp4')
    print("Frames read successfully")

    #Create an instance of the tracker
    tracker = Tracker("models/best.pt")

    print("Tracker created successfully")
    # detections = tracker.detect_frames(frames)
    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl")
    print("Detections done successfully")
    # print(tracks)

    # save cropped image of a player
    # for track_id , player in tracks["players"][0].items():
    #     bbox = player ['bbox']
    #     frame = frames[0]  

    #     #crop bbox from the frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

    #     #save the cropped image
    #     cv2.imwrite(f"out_vid/cropped_img.jpg",cropped_image)

    #     break 

    #Interpolate the ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])


    #Assign player teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks["players"][0])

    for frame_num , player_track in enumerate(tracks["players"]):
        for player_id , track  in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num],track["bbox"],player_id)

            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # Draw the annotations
    output_frames = tracker.draw_annotations(frames, tracks)
    print("Annotations drawn successfully")

    # Save the video
    save_video(output_frames, 'out_vid/train1.avi')
    print("Video saved successfully")

if __name__ == '__main__':
    main()