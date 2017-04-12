import sys
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.all import crop
from moviepy.video.fx.all import resize

if len(sys.argv) == 2:
    path = sys.argv[1]
    clip = VideoFileClip(path)
    cropped = crop(clip, y1=750, y2=1750)
    resized = resize(cropped, newsize=0.5)
    resized.write_videofile('{}_reduced.mp4'.format(path.split(".")[0]), audio=False)
else:
    print("Usage: python reduce_blackfly_video.py [path]")