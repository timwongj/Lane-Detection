import sys
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from os import listdir
from os.path import isfile, join, basename

if len(sys.argv) == 3 or len(sys.argv) == 4:
    path = sys.argv[1]
    files = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and not f.startswith('.')]
    clip = ImageSequenceClip(files, fps=float(sys.argv[2]))
    if len(basename(path)) == 0:
        path = path[:-1]
    if len(sys.argv) == 4:
        clip.write_videofile('{}.mp4'.format(basename(path)), audio=False, fps=float(sys.argv[3]))
    else:
        clip.write_videofile('{}.mp4'.format(basename(path)), audio=False, fps=float(sys.argv[2]))
else:
    print("Usage: python make_video.py [dir] [src_fps]")
    print("Usage: python make_video.py [dir] [src_fps] [dst_fps]")