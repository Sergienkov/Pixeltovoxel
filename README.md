# Pixeltovoxelprojector
Projects motion of pixels to a voxel

## Live voxelized video viewer

Use `voxelliveviewer.py` to open a camera on macOS or play back a video file and
visualize a voxelized version of the frames.

Install dependencies:

```
pip install opencv-python pyvista numpy
```

Examples:

```
python voxelliveviewer.py --list-devices
python voxelliveviewer.py --device 0
python voxelliveviewer.py --file /path/to/video.mp4
```

If you omit `--device` and `--file`, the script will prompt you to choose a camera
index or a file path.
