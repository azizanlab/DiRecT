import os
import skvideo.io


def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_video(filename, video_frames, fps=60, video_format="mp4"):
    assert fps == int(fps), fps
    _make_dir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            "-r": str(int(fps)),
        },
        outputdict={
            "-f": video_format,
            "-pix_fmt": "yuv420p",
        },
    )
