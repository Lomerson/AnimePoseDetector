# -*- coding: utf-8 -*-
"""CLI 入口：AnimePoseDetector。

用法::

    conda run -n torch3d python /home/lomerson/MyProjects/AnimePoseDetector/main.py \
      --input  /home/lomerson/MyProjects/AnimePoseDetector/ref.png \
      --smplx-dir /home/lomerson/MyProjects/models/smplx \
      --smplx-gender NEUTRAL

默认在输入图同目录写出 ``<stem>_annotated.png`` 与 ``<stem>_result.json``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from anime_pose_detector import DEFAULT_SMPLX_DIR, REQUIRED_JOINTS, DERIVED_JOINTS, estimate


def _summary(res: dict) -> str:
    sm = res["smplx"]
    fit = sm["image_to_model_2d"]
    joints = res["joints"]
    total = len(REQUIRED_JOINTS) + len(DERIVED_JOINTS)
    found = sum(1 for j in joints.values() if j.get("xy") is not None)
    eyes = res["eyes"]
    eye_ok = bool(eyes["left"]["center"] and eyes["right"]["center"])
    tm = res["diagnostics"]["timing_ms"]
    lines = [
        "=" * 68,
        f"image        : {res['image_path']}  ({res['image_width']}x{res['image_height']})",
        f"background   : {res['diagnostics']['background']}",
        f"elapsed_ms   : {res['elapsed_ms']}  (load_template {tm['load_template']} ms 稳态: "
        f"mask {tm['mask']} + face/eyes {tm['face_eyes']} + joints {tm['joints']} + fit {tm['fit']})",
        f"method       : {res['method']}",
        f"smplx        : gender={sm['gender']}  {os.path.basename(sm['model_path'])}",
        f"eyes         : {'OK' if eye_ok else 'FAILED'}  "
        f"mid={res['eye_midpoint']}",
        f"face_bbox    : {res['face_bbox']}   cascade={res.get('face_bbox_cascade')}"
        f"   detector={res['diagnostics']['face_detector'].get('name')}"
        f" ({res['diagnostics']['face_detector'].get('detect_ms')} ms)",
        f"joints       : {found}/{total}",
        f"pixels/m     : {fit['scale']:.2f}   rotation={fit['rotation_deg']:.2f} deg   "
        f"t={[round(v, 1) for v in fit['translation_px']]}",
        f"fit rmse     : {fit['rmse_px']:.1f} px  ({fit['n_points']} pts)",
    ]
    pr = res["diagnostics"]["proportion_report"]
    lines.append(
        f"proportions  : shoulder/hip={pr['shoulder_to_hip_ratio']:.2f} "
        f"(template {pr['shoulder_to_hip_ratio_template']:.2f})   "
        f"height/hip={pr['height_over_hip_width']:.2f}"
    )
    if "annotated_path" in res:
        lines.append(f"annotated    : {res['annotated_path']}")
    lines.append(f"json         : {res['result_json_path']}")
    lines.append("=" * 68)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="动漫 T-Pose 轻量 2D 姿态估计（SMPL-X 前 25 关节 + 派生手掌中心）")
    ap.add_argument("--input", required=True, help="输入图片路径")
    ap.add_argument("--smplx-dir", default=DEFAULT_SMPLX_DIR, help="SMPL-X .npz 目录（只读）")
    ap.add_argument("--smplx-gender", default="NEUTRAL",
                    choices=["NEUTRAL", "MALE", "FEMALE"], help="使用哪个 .npz")
    ap.add_argument("--output-dir", default=None, help="输出目录（默认与输入同目录）")
    ap.add_argument("--no-vis", action="store_true", help="不写 *_annotated.png")
    ap.add_argument("--allow-rotation", action="store_true",
                    help="允许 image_to_model_2d 含平面内旋转")
    ap.add_argument("--face-cascade", default=None,
                    help="Anime Face Detector 级联 xml 路径"
                         "（默认自动探测 ./lbpcascade_animeface.xml）")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"[error] 找不到输入文件: {args.input}", file=sys.stderr)
        return 2

    res = estimate(
        args.input,
        smplx_gender=args.smplx_gender,
        smplx_dir=args.smplx_dir,
        output_dir=args.output_dir,
        allow_rotation=args.allow_rotation,
        write_vis=not args.no_vis,
        face_cascade=args.face_cascade,
    )
    print(_summary(res))
    if args.no_vis:
        print(json.dumps({"elapsed_ms": res["elapsed_ms"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
