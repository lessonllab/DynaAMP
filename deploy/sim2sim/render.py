"""Render the recorded MuJoCo states with command and handoff annotations."""

import argparse
import json
from pathlib import Path
import sys

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.runtime import DEFAULT_BUNDLE, G1_SCENE  # noqa: E402


def read_metadata(bundle):
    return json.loads((bundle / "deployment.json").read_text(encoding="utf-8"))


PHASES = {
    "DANCE": "LoopMimic 舞蹈",
    "BLEND": "交接关节目标与 PD",
    "SETTLE": "零速收步",
    "HOLD_BLEND": "切换到站立策略",
    "STAND": "双脚站立 · 可遥控",
    "WALK": "速度指令 · 行走",
    "BRAKE": "指令回零 · 收步",
    "INIT": "SONIC 上电站姿准备",
    "STARTING_CONTROL": "启动 SONIC 控制",
    "PLANNER_START": "启动 SONIC 规划器",
    "LOWERING": "准备双脚着地",
    "IDLE_BEFORE_DANCE": "SONIC 原地待机",
    "DANCE_REQUEST": "遥控器触发舞蹈",
    "RETURN_TO_IDLE": "舞蹈结束 · 交回 SONIC",
    "IDLE": "SONIC 原地待机",
    "FAST_WALK": "SONIC 快走",
    "RUN": "SONIC 跑步",
    "DECELERATE": "SONIC 减速",
    "STOP": "SONIC 停车待机",
}
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def render(trace: Path, output: Path, bundle: Path, tail_seconds: float | None,
           scene: Path | None, start_at: float | None = None) -> int:
    meta = read_metadata(bundle)
    model = mujoco.MjModel.from_xml_path(str(scene or G1_SCENE))
    model.vis.quality.shadowsize = 1024
    model.mat_reflectance[:] = 0.0
    data = mujoco.MjData(model)
    with np.load(trace) as saved:
        poses = saved["qpos"]
        phases = saved["phase"]
        times = saved["time_s"]
        commands = saved["command"] if "command" in saved else None
        body_vx = saved["body_vx"] if "body_vx" in saved else None
        ground_speed = np.linalg.norm(saved["velocity"][:, :2], axis=1)
        feet = saved["contacts"]
    if poses.shape != (len(times), model.nq):
        raise ValueError("Trace and G1 scene dimensions differ")
    begin = start_at if start_at is not None else max(0.0, times[-1] - tail_seconds) if tail_seconds else 0.0
    frames = np.flatnonzero(times > begin)[::2]
    if not len(frames):
        raise ValueError("Start time lies outside the recorded trace")
    fps = round(0.5 / meta["control_dt"])
    size = (960, 672)
    model.vis.global_.offwidth, model.vis.global_.offheight = size
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth = 133.0
    camera.elevation = -13.0
    camera.distance = 2.95 if tail_seconds else 2.75
    camera.lookat[:] = poses[frames[0], :3]
    camera.lookat[2] = 0.65

    output.parent.mkdir(parents=True, exist_ok=True)
    video = imageio_ffmpeg.write_frames(
        str(output), size, fps=fps, codec="libx264", pix_fmt_out="yuv420p",
        output_params=["-preset", "veryfast", "-crf", "19", "-movflags", "+faststart"],
    )
    title = ImageFont.truetype(FONT, 27, index=2)
    label = ImageFont.truetype(FONT, 18, index=2)
    video.send(None)
    try:
        with mujoco.Renderer(model, width=size[0], height=size[1]) as renderer:
            for i in frames:
                # Replay measured poses of the same physics episode.
                data.qpos[:] = poses[i]
                mujoco.mj_forward(model, data)
                camera.lookat[:2] += 0.055 * (poses[i, :2] - camera.lookat[:2])
                renderer.update_scene(data, camera=camera, scene_option=option)
                img = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(img, "RGBA")
                draw.rectangle((0, 0, size[0], 64), fill=(11, 19, 28, 224))
                phase = "SONIC 慢走" if commands is None and phases[i] == "WALK" else PHASES.get(
                    phases[i], "仿真停止"
                )
                draw.text((22, 14), phase, font=title, fill="white")
                draw.text((size[0] - 22, 20), f"仿真时间  {times[i]:06.2f} s",
                          anchor="ra", font=label, fill=(219, 235, 246, 255))
                draw.rectangle((0, size[1] - 53, size[0], size[1]), fill=(11, 19, 28, 224))
                if commands is None:
                    command_caption = "原生 SONIC · MuJoCo / DDS"
                    speed_caption = f"实测水平速度  {ground_speed[i]:.2f} m/s"
                else:
                    vx, vy, yaw = commands[i]
                    command_caption = f"指令  vx {vx:+.2f}   vy {vy:+.2f} m/s   ω {yaw:+.2f} rad/s"
                    speed_caption = f"实测前向速度  {body_vx[i]:+.2f} m/s"
                draw.text((22, size[1] - 40), command_caption, font=label, fill=(219, 235, 246, 255))
                draw.text((size[0] - 24, size[1] - 40), speed_caption,
                          anchor="ra", font=label, fill=(219, 235, 246, 255))
                for row, down in enumerate(feet[i]):
                    x = 540 + 55 * row
                    y = size[1] - 29
                    color = (94, 225, 170, 255) if down else (130, 143, 153, 255)
                    draw.ellipse((x, y - 4, x + 8, y + 4), fill=color)
                    draw.text((x + 13, size[1] - 40), "左" if row == 0 else "右",
                              font=label, fill=(219, 235, 246, 255))
                video.send(np.asarray(img))
    finally:
        video.close()
    return len(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=G1_SCENE,
                        help="Use the same Unitree scene as the recorded trace")
    section = parser.add_mutually_exclusive_group()
    section.add_argument("--tail-seconds", type=float)
    section.add_argument("--start-at", type=float)
    args = parser.parse_args()
    if args.tail_seconds is not None and args.tail_seconds <= 0:
        parser.error("--tail-seconds must be positive")
    frames = render(args.trace, args.output, args.bundle, args.tail_seconds, args.scene, args.start_at)
    print(f"Rendered {frames} frames to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
