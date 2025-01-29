from utils import read_video, save_video

def main():
    # Read the video
    frames = read_video('inp_vid/train1.mp4')

    # Save the video
    save_video(frames, 'out_vid/train1.avi')

if __name__ == '__main__':
    main()