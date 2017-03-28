import sys
from moviepy.video.io.VideoFileClip import VideoFileClip

if len(sys.argv) == 3:
    path = sys.argv[1]
    clip = VideoFileClip(path)
    clip.write_videofile('{}_{}fps.mp4'.format(path.split(".")[0], sys.argv[2]), audio=False, fps=float(sys.argv[2]))
else:
    print("Usage: python convert_fps.py [path] [fps]")