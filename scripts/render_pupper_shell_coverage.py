"""Render the retained and removed Pupper shell ranges in compact pose."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from curl_robot_2d.model import _angular_distance, _point_segment_distance
from curl_robot_2d.parameters import PUPPER_ORIGINAL_SHELL_PARAMETERS as P
from curl_robot_2d.pupper_shell_geometry import solve_compact_geometry


OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results/pupper_r127p5_shank_shell_trim_three_stage_cem"
    / "shell_coverage_compact.png"
)


def main() -> None:
    solution = solve_compact_geometry(P.pupper_shell_design)
    center = (0.0, -solution.shell_center_below_hip)
    front_hip = (P.hip_half_span, 0.0)
    front_knee = (solution.knee_x, -solution.knee_below_hip)
    front_foot = (solution.foot_x, -solution.foot_below_hip)
    rear_hip = (-front_hip[0], 0.0)
    rear_knee = (-front_knee[0], front_knee[1])
    rear_foot = (-front_foot[0], front_foot[1])

    def polar(point: tuple[float, float]) -> float:
        return math.atan2(point[1] - center[1], point[0] - center[0])

    anchors = {
        "torso": math.pi / 2.0,
        "front thigh": polar(((front_hip[0] + front_knee[0]) / 2,
                               (front_hip[1] + front_knee[1]) / 2)),
        "front shank": polar(((front_knee[0] + front_foot[0]) / 2,
                               (front_knee[1] + front_foot[1]) / 2)),
        "rear shank": polar(((rear_knee[0] + rear_foot[0]) / 2,
                              (rear_knee[1] + rear_foot[1]) / 2)),
        "rear thigh": polar(((rear_hip[0] + rear_knee[0]) / 2,
                              (rear_hip[1] + rear_knee[1]) / 2)),
    }
    feet = {"front shank": front_foot, "rear shank": rear_foot}
    colors = {
        "torso": "#29b6f6",
        "front thigh": "#ff9800",
        "front shank": "#ffeb3b",
        "rear shank": "#7ee787",
        "rear thigh": "#ab47bc",
    }

    image = Image.new("RGB", (1600, 1500), "#101722")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 34)
    small = ImageFont.truetype("arial.ttf", 27)
    scale = 4300.0
    origin = (800.0, 650.0)

    def px(point: tuple[float, float]) -> tuple[float, float]:
        return origin[0] + scale * point[0], origin[1] - scale * point[1]

    def circle(point, radius, fill, outline="#e6edf3", width=4):
        x, y = px(point)
        r = scale * radius
        draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline=outline, width=width)

    # Kinematic skeleton and protection envelopes.
    draw.line((*px(rear_hip), *px(front_hip)), fill="#8b949e", width=9)
    for hip, knee, foot in (
        (front_hip, front_knee, front_foot),
        (rear_hip, rear_knee, rear_foot),
    ):
        draw.line((*px(hip), *px(knee)), fill="#b36b35", width=18)
        draw.line((*px(knee), *px(foot)), fill="#d6a128", width=18)
        circle(hip, P.motor_radius, "#555d68")
        circle(knee, P.motor_radius, "#555d68")
        circle(foot, P.foot_radius, "#f0f3f6")

    kept = 0
    removed = 0
    radius = P.shell_centerline_radius
    for index in range(P.shell_segments_full_circle):
        angle_a = 2 * math.pi * index / P.shell_segments_full_circle
        angle_b = 2 * math.pi * (index + 1) / P.shell_segments_full_circle
        middle = (angle_a + angle_b) / 2
        body = min(anchors, key=lambda name: _angular_distance(middle, anchors[name]))
        point_a = (center[0] + radius * math.cos(angle_a),
                   center[1] + radius * math.sin(angle_a))
        point_b = (center[0] + radius * math.cos(angle_b),
                   center[1] + radius * math.sin(angle_b))
        foot = feet.get(body)
        is_removed = foot is not None and _point_segment_distance(
            foot, point_a, point_b
        ) < P.foot_radius + P.shell_capsule_radius + P.pupper_shank_shell_foot_clearance
        if is_removed:
            removed += 1
            # Red dashed centerline indicates geometry absent from MJCF.
            for part in (0.06, 0.31, 0.56, 0.81):
                aa = angle_a + part * (angle_b - angle_a)
                bb = angle_a + (part + 0.13) * (angle_b - angle_a)
                draw.line((*px((center[0]+radius*math.cos(aa), center[1]+radius*math.sin(aa))),
                           *px((center[0]+radius*math.cos(bb), center[1]+radius*math.sin(bb)))),
                          fill="#ff4d4f", width=15)
        else:
            kept += 1
            draw.line((*px(point_a), *px(point_b)), fill=colors[body], width=18)

    circle(center, 0.003, "#ff4d4f", outline="#ff4d4f", width=1)
    cx, cy = px(center)
    draw.text((cx + 14, cy - 18), "shell center", font=small, fill="#ff8f8f")

    draw.text((55, 35), "Current shell coverage - compact pose", font=font, fill="white")
    draw.text((55, 84), f"R = 127.5 mm | 48 nominal segments | {kept} retained | {removed} removed",
              font=small, fill="#c9d1d9")
    opening_deg = 360.0 * removed / P.shell_segments_full_circle
    draw.text((55, 123), f"Bottom opening: {opening_deg:.0f} deg (225 deg to 315 deg)",
              font=small, fill="#ff8f8f")

    y = 1260
    legend = [
        ("#29b6f6", "torso shell: 12 segments"),
        ("#ff9800", "front thigh shell: 10 segments"),
        ("#ffeb3b", "front shank shell: 2 segments"),
        ("#7ee787", "rear shank shell: 2 segments"),
        ("#ab47bc", "rear thigh shell: 10 segments"),
        ("#ff4d4f", "red dashed: removed near-foot shell: 12 segments"),
    ]
    for col, label in legend:
        draw.line((70, y+16, 130, y+16), fill=col, width=13)
        draw.text((150, y), label, font=small, fill="#e6edf3")
        y += 36

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
