## make_video.py

- creates a video out of a sequence of images
- specify directory and source fps
- target fps is optional

Usage:

```
python make_video.py [dir] [src_fps]
python make_video.py [dir] [src_fps] [dst_fps]
```

Examples:
```
python make_video.py ~/Desktop/images 15
python make_video.py ~/Desktop/images 15 5
```

## convert_fps.py

- converts frame rate of video
- specify path and target fps

Usage:

```
python convert_fps.py [path] [fps]
```

Example:
```
python convert_fps.py ~/Desktop/video.mp4 10
```