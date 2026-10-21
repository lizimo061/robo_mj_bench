"""Render README GIFs on the trusted operator side; these are not LLM scores.

Seven tasks show small actuator-driven arm motions in their initial scenes.
An optional existing button integration video supplies the solved button clip.
"""

import argparse
import io
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from robo_mj_bench.protocol import HOME_JOINTS
from robo_mj_bench.tasks.registry import TASK_NAMES, create_simulation


def render_scene(name, output, seed):
    simulation = create_simulation(name, seed)
    frames = []
    try:
        for index in range(32):
            target = list(HOME_JOINTS)
            target[0] += .06 * math.sin(2 * math.pi * index / 32)
            simulation.move_joints(target, .125)
            canvas = Image.new("RGB", (640, 284), "#101620")
            for camera, x in (("front", 0), ("overhead", 320)):
                with Image.open(io.BytesIO(simulation.observe(camera))) as image:
                    canvas.paste(image.resize((320, 240), Image.Resampling.LANCZOS), (x, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 244), name.replace("_", " ").title(),
                      font=ImageFont.load_default(size=16), fill="white")
            draw.text((8, 264), "Scene preview | scripted arm motion | task not solved",
                      font=ImageFont.load_default(size=12), fill="#bdcbdc")
            frames.append(canvas)
        # One shared palette prevents distracting color changes across frames.
        palette = frames[0].quantize(colors=128)
        frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        frames[0].save(output, save_all=True, append_images=frames[1:],
                       duration=125, loop=0, optimize=True, disposal=2)
    finally:
        simulation.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/media"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--button-video", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in TASK_NAMES:
        output = args.output / f"{name}.gif"
        if name == "button" and args.button_video is not None:
            # Preserve the original recording's diagnostic and result labels.
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(args.button_video), "-filter_complex",
                "fps=8,scale=640:-1:flags=lanczos,split[a][b];"
                "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
                "-loop", "0", str(output),
            ], check=True)
        else:
            render_scene(name, output, args.seed)
        print(f"{output}: {output.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
