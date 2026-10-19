# -*- coding: utf-8 -*-
"""AnimePoseDetector —— 动漫人物 T-Pose 轻量 2D 姿态估计（SMPL-X 语义对齐）。

设计目标
--------
* 单张图 **秒级以内**（实测稳态 < 200 ms, CPU only）。
* **不使用任何神经网络 / 模型权重**，只依赖 ``numpy``, ``scipy``, ``opencv-python-headless``, ``Pillow``。
  不 import torch，因此不会触碰/影响本机的 PyTorch3D 环境。
* 使用本地 SMPL-X 中性模型 ``.npz`` 的 ``v_template`` + ``J_regressor`` 作为 **几何先验**
  （关节相对比例、T-Pose 骨架叠加、2D 相似变换初值），**只读**，不写回、不拷贝。
* 输出坐标一律是 **原图像素坐标**：原点左上，``x`` 向右，``y`` 向下。

坐标 / 投影约定（``image_to_model_2d``）
---------------------------------------
SMPL-X 模型坐标：``X`` = 人物左右（**左为正**），``Y`` = 向上，``Z`` = 前后（前为正）。
标准 T-Pose 下 ``|Z| < 0.13 m``，因此采用 **正交投影并丢弃 Z**：

    m = [X, -Y]                                  # 单位: 米, y 轴翻转为图像方向(向下)
    p_img = scale * R(rotation_deg) @ m + translation_px   # 单位: 像素

* ``scale``          : 像素/米 (pixels per meter)，即 ``pixels_per_meter``。
* ``rotation_deg``   : 图像平面内旋转角（度）。按图像坐标系（x 右 / y 下）逆时针为正；
                       默认 ``allow_rotation=False`` 时恒为 0。
* ``translation_px`` : 模型原点 (pelvis 投影点) 在图像中的像素位置 ``[tx, ty]``。

该变换方向是 **model → image**（把模板骨架投到图上），字段名沿用任务约定
``image_to_model_2d``；其逆变换可由 ``scale``/``rotation_deg``/``translation_px`` 直接求出：
``m = R(-rot) (p_img - t) / scale``。

左右约定
--------
SMPL-X 中 ``X+`` 是人物**自身**的左侧。对正面朝向镜头的角色，人物的左手出现在**图像右侧**。
本模块所有 ``left_*`` / ``right_*`` 关节、以及 ``eyes.left`` / ``eyes.right``
**一律是人物自身左右**（与 SMPL-X 语义一致），因此：

    left_shoulder.xy[0] > right_shoulder.xy[0]     # 人物左肩在图像右侧
    eyes.left.center[0] > eyes.right.center[0]

关节索引（SMPL-X 官方 55 关节顺序，环境内 ``smplx/joint_names.py`` 核对过）
--------------------------------------------------------------------------
0 pelvis         1 left_hip       2 right_hip      3 spine1
4 left_knee      5 right_knee     6 spine2         7 left_ankle
8 right_ankle    9 spine3        10 left_foot     11 right_foot
12 neck         13 left_collar   14 right_collar  15 head
16 left_shoulder 17 right_shoulder 18 left_elbow  19 right_elbow
20 left_wrist   21 right_wrist   22 jaw          23 left_eye_smplhf
24 right_eye_smplhf              25..39 left hand fingers
40..54 right hand fingers

注意：SMPL-X **没有** 名为 ``left_hand`` 的官方关节（22 = jaw, 23/24 = 双眼）。
本模块仍按任务要求输出 ``left_hand`` / ``right_hand``（手掌中心），但其
``smplx_index`` 为 ``None``，并用 ``derived_from`` 说明它由手腕 + 手指关节几何推导，
避免与官方索引冲突。

对外 API
--------
``load_smplx_template`` / ``detect_person_mask`` / ``detect_face_roi`` / ``detect_eyes``
``estimate_joints`` / ``fit_similarity_2d`` / ``visualize`` / ``estimate``
"""

from __future__ import annotations

import json
import math
import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

__all__ = [
    "SMPLX_JOINT_NAMES",
    "SMPLX_INDEX",
    "REQUIRED_JOINTS",
    "load_smplx_template",
    "detect_person_mask",
    "load_anime_face_cascade",
    "detect_anime_face",
    "detect_face_roi",
    "detect_eyes",
    "estimate_joints",
    "fit_similarity_2d",
    "visualize",
    "estimate",
]

DEFAULT_SMPLX_DIR = "/home/lomerson/MyProjects/models/smplx"

# ---------------------------------------------------------------------------
# 常量：SMPL-X 官方 55 关节名 / 索引
# ---------------------------------------------------------------------------
SMPLX_JOINT_NAMES: List[str] = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "jaw", "left_eye_smplhf", "right_eye_smplhf",
    # 左手手指 25..39
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    # 右手手指 40..54
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
]
SMPLX_INDEX: Dict[str, int] = {n: i for i, n in enumerate(SMPLX_JOINT_NAMES)}

#: 必须输出的 2D 关节（SMPL-X 官方 0..24）
REQUIRED_JOINTS: List[Tuple[int, str]] = [(i, SMPLX_JOINT_NAMES[i]) for i in range(25)]
#: 任务要求但 SMPL-X 无对应索引的派生关节
DERIVED_JOINTS: List[str] = ["left_hand", "right_hand"]

#: SMPL-X 标准骨骼连线（parent -> child，取自 npz 的 kintree_table）
_SMPLX_PARENTS: List[int] = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
    13, 14, 16, 17, 18, 19, 15, 15, 15,
    20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
]
_BODY_EDGES: List[Tuple[int, int]] = [
    (p, c) for c, p in enumerate(_SMPLX_PARENTS) if 0 <= p < 25 and c < 25
]
_HAND_EDGES: List[Tuple[str, str]] = [
    ("left_wrist", "left_hand"), ("right_wrist", "right_hand"),
]

# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def _runs(row_bool: np.ndarray) -> List[Tuple[int, int]]:
    """返回一维 bool 数组中每段连续 True 的 (start, end_inclusive)。"""
    idx = np.flatnonzero(row_bool)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], brk + 1)).astype(int)
    ends = np.concatenate((brk, [idx.size - 1])).astype(int)
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def _otsu(values: np.ndarray, nbins: int = 256) -> float:
    """一维 Otsu 阈值（纯 numpy 实现），返回阈值（与输入同量纲）。"""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        return lo
    hist, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) * 0.5
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * centers)
    s1 = s0[-1] - s0
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = s0 / w0
        mu1 = s1 / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = -1.0
    k = int(np.argmax(between))
    return float(centers[k])


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """把 HxWx3 或 HxWx4 图像转换为 RGB uint8（含 alpha 时合成到白底）。"""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 4:
        rgb = arr[..., :3].astype(np.float32)
        a = arr[..., 3:4].astype(np.float32) / 255.0
        rgb = rgb * a + 255.0 * (1.0 - a)
        return np.clip(rgb, 0, 255).astype(np.uint8)
    return arr[..., :3].astype(np.uint8)


def _to_bool(mask: Any) -> np.ndarray:
    m = np.asarray(mask)
    if m.dtype == bool:
        return m
    return m > 127


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """保留最大连通域（4 连通足够，剪影是连续的）。"""
    m8 = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    return lab == k


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """形态学清理 + 填洞 + 最大连通域。"""
    m = mask.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    mb = m > 0
    mb = ndi.binary_fill_holes(mb)
    mb = _largest_component(mb)
    mb = ndi.binary_fill_holes(mb)
    return mb


def _row_extents(mask: np.ndarray):
    """逐行计算 xmin / xmax / 像素数 / 跨度（无效行为 -1）。"""
    h, w = mask.shape
    has = mask.any(axis=1)
    xmin = np.where(has, np.argmax(mask, axis=1), -1)
    rev = mask[:, ::-1]
    xmax = np.where(has, w - 1 - np.argmax(rev, axis=1), -1)
    counts = mask.sum(axis=1)
    ext = np.where(has, xmax - xmin + 1, 0)
    return xmin.astype(int), xmax.astype(int), counts.astype(int), ext.astype(int)


def _background_color(rgb: np.ndarray, ring: int) -> np.ndarray:
    """用图像外圈像素估计背景色（中位数，抗人物触边）。"""
    h, w = rgb.shape[:2]
    ring = max(1, min(ring, h // 4, w // 4))
    border = np.concatenate([
        rgb[:ring].reshape(-1, 3),
        rgb[-ring:].reshape(-1, 3),
        rgb[:, :ring].reshape(-1, 3),
        rgb[:, -ring:].reshape(-1, 3),
    ])
    return np.median(border, axis=0)


# ---------------------------------------------------------------------------
# 1. SMPL-X 模板
# ---------------------------------------------------------------------------

def _npz_path_for(gender: str, smplx_dir: str) -> Tuple[str, str]:
    """探测 ``SMPLX_<GENDER>.npz``，返回 (path, gender)；找不到时回退 NEUTRAL。"""
    g = str(gender).upper().strip()
    cand = os.path.join(smplx_dir, f"SMPLX_{g}.npz")
    if os.path.isfile(cand):
        return cand, g
    for fallback in ("NEUTRAL", "MALE", "FEMALE"):
        p = os.path.join(smplx_dir, f"SMPLX_{fallback}.npz")
        if os.path.isfile(p):
            return p, fallback
    raise FileNotFoundError(
        f"在 {smplx_dir!r} 下找不到 SMPLX_*.npz（只读取 .npz，不使用 .pkl）"
    )


@lru_cache(maxsize=8)
def _load_template_cached(model_path: str):
    """读取 npz 并计算 J_template = J_regressor @ v_template（只读，不写回）。"""
    d = np.load(model_path, allow_pickle=False)
    if "v_template" not in d.files or "J_regressor" not in d.files:
        raise KeyError(f"{model_path} 缺少 v_template / J_regressor")
    jr = d["J_regressor"]
    if hasattr(jr, "toarray"):          # 防御：若将来换成 scipy sparse
        jr = jr.toarray()
    V = np.asarray(d["v_template"], dtype=np.float64)     # (10475, 3)
    Jr = np.asarray(jr, dtype=np.float64)                 # (55, 10475) dense
    J = Jr @ V                                            # (55, 3) 米
    return J.astype(np.float64)


def load_smplx_template(gender: str = "NEUTRAL",
                        smplx_dir: str = DEFAULT_SMPLX_DIR
                        ) -> Tuple[np.ndarray, List[str], List[int]]:
    """加载本地 SMPL-X 模型，返回标准 T-Pose 的 3D 关节。

    只使用 ``.npz`` 中的 ``v_template`` (10475x3) 与 ``J_regressor`` (55x10475)，
    ``J_template = J_regressor @ v_template``。**不使用 .pkl、不安装 chumpy**。
    不写回、不拷贝 ``models/smplx`` 下的任何文件。

    Parameters
    ----------
    gender : {"NEUTRAL", "MALE", "FEMALE"}
        优先使用 ``SMPLX_<GENDER>.npz``；不存在时回退到 NEUTRAL 并记录在实际返回的
        ``model_path``（见 :func:`estimate` 结果里的 ``smplx.model_path``）。
    smplx_dir : str
        模型目录，默认 ``/home/lomerson/MyProjects/models/smplx``。

    Returns
    -------
    (J_template, joint_names, joint_indices)
        ``J_template``: (55, 3) float64，单位米，模型坐标 (X 左正 / Y 上正 / Z 前正)；
        ``joint_names``: SMPL-X 官方 55 关节名；
        ``joint_indices``: ``list(range(55))``。
    """
    path, _ = _npz_path_for(gender, smplx_dir)
    J = _load_template_cached(path)
    return J, list(SMPLX_JOINT_NAMES), list(range(len(SMPLX_JOINT_NAMES)))


def _template_joint(J: np.ndarray, names: Sequence[str], name: str) -> np.ndarray:
    """按名字取模板关节（模板行号由 ``names`` 决定，容忍不同排列）。"""
    i = list(names).index(name) if name in names else SMPLX_INDEX[name]
    return np.asarray(J, dtype=np.float64)[i]


# ---------------------------------------------------------------------------
# 2. 人物剪影
# ---------------------------------------------------------------------------

def detect_person_mask(image: np.ndarray) -> np.ndarray:
    """分割人物剪影，返回 0/255 ``uint8`` mask。

    支持三种输入：
    1. **透明背景**（RGBA / LA）：alpha < 250 的像素占比 > 0.5% 时直接用 alpha；
    2. **纯白 / 任意纯色背景**：用图像外圈像素的中位数估计背景色 ``bg``，
       以 ``||p - bg||`` 为距离图，用 Otsu 自动定阈值（下限 12）分割；
    3. 若 Otsu 结果过于极端（几乎全图或几乎为空），回退到固定阈值 18。
    """
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        a = arr[..., 3]
        if float((a < 250).mean()) > 0.005:
            m = a >= 128
            return (_clean_mask(m) * 255).astype(np.uint8)

    rgb = _to_rgb(arr).astype(np.int16)
    h, w = rgb.shape[:2]
    ring = max(2, int(round(min(h, w) * 0.01)))
    bg = _background_color(rgb.astype(np.uint8), ring=ring)
    dist = np.sqrt(((rgb - bg.reshape(1, 1, 3)) ** 2).sum(axis=2)).astype(np.float32)

    # 背景是纯色 => 用“外圈噪声”自适应阈值（比 Otsu 稳：Otsu 会把浅色皮肤/衣服切掉）
    border = np.concatenate([
        dist[:ring].ravel(), dist[-ring:].ravel(),
        dist[:, :ring].ravel(), dist[:, -ring:].ravel(),
    ])
    med = float(np.median(border))
    mad = float(np.median(np.abs(border - med)))
    sigma = 1.4826 * mad
    thr = max(8.0, med + 6.0 * sigma)

    m = dist > thr
    frac = float(m.mean())
    if frac < 0.01 or frac > 0.97:
        # 退化（渐变背景 / 人物占满画面）-> Otsu 兜底
        thr_o = max(_otsu(dist), 10.0)
        m = dist > thr_o
        frac = float(m.mean())
        if frac < 0.005:
            m = dist > 10.0
    return (_clean_mask(m) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. 脸部 ROI
# ---------------------------------------------------------------------------

def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    """肤色近似分割（针对动漫平涂皮肤，偏桃子色）。"""
    R = rgb[..., 0].astype(np.int16)
    G = rgb[..., 1].astype(np.int16)
    B = rgb[..., 2].astype(np.int16)
    mx = rgb.max(axis=2).astype(np.int16)
    mn = rgb.min(axis=2).astype(np.int16)
    sat = (mx - mn)
    m = (
        (R > 185) & (G > 165) & (B > 140)
        & ((R - B) >= 12) & ((R - G) >= 4)
        & (sat >= 10) & (sat <= 110)
        & ~((R > 250) & (G > 250) & (B > 250))
    )
    m8 = m.astype(np.uint8)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return m8.astype(bool)


# ---------------------------------------------------------------------------
# 3b. Anime Face Detector（lbpcascade_animeface, MIT, OpenCV LBP 级联）
# ---------------------------------------------------------------------------

CASCADE_FILENAME = "lbpcascade_animeface.xml"


def _default_cascade_path() -> Optional[str]:
    """按优先级探测动漫脸级联：显式 env > 模块同目录 > 当前工作目录。"""
    cands: List[str] = []
    env = os.environ.get("ANIME_FACE_CASCADE")
    if env:
        cands.append(env)
    if "__file__" in globals():
        cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), CASCADE_FILENAME))
    cands.append(os.path.join(os.getcwd(), CASCADE_FILENAME))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


@lru_cache(maxsize=4)
def _load_cascade_cached(path: str):
    c = cv2.CascadeClassifier(path)
    if c.empty():
        return None
    # 预热：OpenCV 的级联在第一次 detectMultiScale 时做内部初始化（约 10~20 ms），
    # 在这里一次性付掉，之后每次检测的耗时才是干净的稳态值。
    c.detectMultiScale(np.zeros((64, 64), np.uint8), scaleFactor=1.1,
                       minNeighbors=3, minSize=(24, 24))
    return c


def load_anime_face_cascade(cascade_path: Optional[str] = None):
    """加载 Anime Face Detector 级联（``lbpcascade_animeface.xml``，MIT）。

    纯 OpenCV 的 LBP 级联：无神经网络、无 Python 新依赖、约 241 KB。
    找不到文件时返回 ``None``（调用方自动回退到肤色启发式）。

    Returns
    -------
    cv2.CascadeClassifier | None
    """
    path = cascade_path or _default_cascade_path()
    if not path or not os.path.isfile(path):
        return None
    try:
        return _load_cascade_cached(os.path.abspath(path))
    except Exception:
        return None


def detect_anime_face(image: np.ndarray,
                      cascade_path: Optional[str] = None,
                      *,
                      person_mask: Optional[np.ndarray] = None,
                      search_region: Optional[Tuple[int, int, int, int]] = None,
                      scale_factor: float = 1.1,
                      min_neighbors: int = 5,
                      min_size: int = 24) -> Optional[Dict[str, Any]]:
    """用 Anime Face Detector（LBP 级联）检测动漫脸，返回最佳一框。

    速度优化：给出 ``person_mask`` 时只在「人物头部区域」（剪影上部 ~52%）内检测，
    实测比全图快约 40%；若头部区域内没检测到，再回退到全图重试一次。

    Parameters
    ----------
    image : np.ndarray
        RGB/RGBA ``uint8``；也接受灰度图。
    cascade_path : str | None
        级联 xml 路径；``None`` 时自动在模块目录 / 当前目录找。
    person_mask : np.ndarray | None
        人物剪影；既用于限定搜索区域，也用于剔除落在剪影外的误检。
    search_region : (x, y, w, h) | None
        显式指定搜索区域（优先于由 ``person_mask`` 推导的头部区域）。
    scale_factor, min_neighbors, min_size
        直通 ``cv2.CascadeClassifier`` 的参数。

    Returns
    -------
    dict | None
        ``{'bbox': [x, y, w, h], 'weight': float, 'n_candidates': int,
        'all_bboxes': [[x,y,w,h], ...], 'cascade_path': str,
        'init_ms': float, 'detect_ms': float, 'search_region': [...] }``；
        级联不可用或没有检测到时返回 ``None``。
    """
    t_load0 = time.perf_counter()
    cascade = load_anime_face_cascade(cascade_path)
    init_ms = (time.perf_counter() - t_load0) * 1000.0
    if cascade is None:
        return None

    arr = np.asarray(image)
    if arr.ndim == 2:
        gray_full = arr.astype(np.uint8)
    else:
        gray_full = cv2.cvtColor(_to_rgb(arr), cv2.COLOR_RGB2GRAY)
    h, w = gray_full.shape[:2]

    mask = _to_bool(person_mask) if person_mask is not None else None
    regions: List[Optional[Tuple[int, int, int, int]]] = []
    if search_region is not None:
        regions = [tuple(int(v) for v in search_region), None]
    elif mask is not None:
        ys, xs = np.nonzero(mask)
        if ys.size:
            y0, y1 = int(ys.min()), int(ys.max())
            span = max(1, y1 - y0)
            regions = [(0, max(0, y0 - int(0.02 * span)), w,
                        min(h, y0 + int(0.52 * span)) - max(0, y0 - int(0.02 * span))), None]
    if not regions:
        regions = [None]

    t_det0 = time.perf_counter()
    found = None
    used_region = None
    for reg in regions:
        if reg is None:
            rx, ry, rw, rh = 0, 0, w, h
        else:
            rx, ry, rw, rh = reg
            rx, ry = max(0, rx), max(0, ry)
            rw, rh = min(rw, w - rx), min(rh, h - ry)
            if rw < 32 or rh < 32:
                continue
        sub = gray_full[ry:ry + rh, rx:rx + rw]
        sub = cv2.equalizeHist(sub)     # 官方示例做法，低对比图更稳
        try:
            rects, _levels, weights = cascade.detectMultiScale3(
                sub, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                minSize=(min_size, min_size), outputRejectLevels=True)
            weights = np.asarray(weights).ravel()
        except Exception:               # 极老版本兜底
            rects = cascade.detectMultiScale(sub, scaleFactor=scale_factor,
                                             minNeighbors=min_neighbors,
                                             minSize=(min_size, min_size))
            weights = np.ones(len(rects))
        if len(rects) == 0:
            continue
        cands: List[Tuple[float, List[int], float]] = []
        for k, (x, y, bw, bh) in enumerate(np.asarray(rects).reshape(-1, 4)):
            x, y = int(x) + rx, int(y) + ry
            bw, bh = int(bw), int(bh)
            if bw <= 2 or bh <= 2:
                continue
            if mask is not None:
                cx, cy = int(x + bw / 2), int(y + bh / 2)
                if not (0 <= cx < w and 0 <= cy < h and mask[min(cy, h - 1), min(cx, w - 1)]):
                    continue
            wt = float(weights[k]) if k < weights.size else 1.0
            cands.append((float(bw * bh) * (1.0 + 0.02 * max(0.0, wt)), [x, y, bw, bh], wt))
        if cands:
            cands.sort(key=lambda t: -t[0])
            found = cands
            used_region = [rx, ry, rw, rh]
            break
    detect_ms = (time.perf_counter() - t_det0) * 1000.0
    if not found:
        return None
    _, best_bbox, best_w = found[0]
    return {
        "bbox": best_bbox,
        "weight": best_w,
        "n_candidates": len(found),
        "all_bboxes": [c[1] for c in found],
        "cascade_path": str(cascade_path or _default_cascade_path()),
        "init_ms": round(init_ms, 2),
        "detect_ms": round(detect_ms, 2),
        "search_region": used_region,
    }


def _tighten_face_bbox(comp: np.ndarray) -> Tuple[int, int, int, int]:
    """把「脸+脖子」的肤色连通域收窄成「脸」的 bbox（相对 comp 左上角）。

    做法：逐行像素数当作宽度，找上半部最宽行（脸颊），再往下找第一次明显收窄（下巴）。
    """
    widths = comp.sum(axis=1)
    rows_c = np.flatnonzero(widths > 0)
    if rows_c.size == 0:
        return 0, 0, comp.shape[1], comp.shape[0]
    # 用“宽度阈值”定上沿：忽略深入头发里的细长突起（常见于重采样/抗锯齿后）
    # 上沿就取连通域自身的第一行：动漫脸在眼睛高度往往被刘海/鬓发切成窄条，
    # 用“宽度阈值”收紧会直接把眼睛切掉。发丝/重采样噪声由下面的级联框先验约束。
    top = int(rows_c[0])
    upper_rows = rows_c[rows_c <= top + 0.55 * (rows_c[-1] - top + 1)]
    peak = int(upper_rows[np.argmax(widths[upper_rows])]) if upper_rows.size else top
    wmax = float(widths[peak])
    bottom = int(rows_c[-1])
    for yv in range(peak, int(rows_c[-1]) + 1):
        if widths[yv] < 0.55 * wmax:
            bottom = int(yv)
            break
    sub = comp[top: bottom + 1]
    cols = np.flatnonzero(sub.any(axis=0))
    if cols.size == 0:
        return 0, 0, comp.shape[1], comp.shape[0]
    return int(cols[0]), top, int(cols[-1] - cols[0] + 1), int(bottom - top + 1)


def _face_bbox_from_skin(image: np.ndarray, person_mask: np.ndarray,
                         window: Optional[Tuple[int, int, int, int]] = None
                         ) -> Optional[Tuple[int, int, int, int]]:
    """肤色 + 连通域找脸（启发式，同时用作级联结果的细化与兜底）。

    ``window`` 给定时**只在窗口内做肤色分割**（先裁剪，约 50× 更快），
    用于把级联给出的「头部框」细化成「脸部框」。
    """
    rgb = _to_rgb(image)
    mask = _to_bool(person_mask)
    h, w = mask.shape

    if window is not None:
        wx, wy, ww, wh = [int(round(v)) for v in window]
        wx, wy = max(0, wx), max(0, wy)
        ww, wh = min(ww, w - wx), min(wh, h - wy)
        if ww < 8 or wh < 8:
            return None
        sub_rgb = rgb[wy:wy + wh, wx:wx + ww]
        sub_mask = mask[wy:wy + wh, wx:wx + ww]
        skin = _skin_mask(sub_rgb) & sub_mask
        n, lab, stats, _cent = cv2.connectedComponentsWithStats(skin.astype(np.uint8), connectivity=8)
        best, best_score = None, -1.0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 60:
                continue
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            ar = bw / max(1.0, float(bh))
            if not (0.30 <= ar <= 2.5):
                continue
            if 0.25 * bw < 12 or 0.25 * bh < 12:
                continue
            score = math.log(area)
            if score > best_score:
                best_score, best = score, i
        if best is None:
            return None
        comp = (lab == best)
        fx, fy, fw, fh = _tighten_face_bbox(comp)
        fx, fy = fx + wx, fy + wy
    else:
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            return None
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        figure_h = max(1, y1 - y0 + 1)
        xmin, xmax, _, _ext = _row_extents(mask)
        rows = np.arange(y0, y1 + 1)
        axis = float(np.median((xmin[rows] + xmax[rows]) / 2.0))

        search = np.zeros_like(mask)
        search[int(y0 + 0.02 * figure_h):int(y0 + 0.50 * figure_h), :] = True
        skin = _skin_mask(rgb) & mask & search
        n, lab, stats, cent = cv2.connectedComponentsWithStats(skin.astype(np.uint8), connectivity=8)
        best, best_score = None, -1.0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < max(60, int(0.0015 * w * h)):
                continue
            cx, cy = cent[i]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if bh < 0.02 * figure_h or bw < 0.02 * w:
                continue
            ar = bw / max(1.0, bh)
            if not (0.35 <= ar <= 2.2):
                continue
            d_axis = abs(cx - axis) / max(1.0, 0.5 * (x1 - x0 + 1))
            rel_y = (cy - y0) / figure_h
            score = math.log(area) - 2.2 * d_axis - 3.0 * max(0.0, rel_y - 0.30)
            if score > best_score:
                best_score, best = score, i
        if best is None:
            return None
        comp = (lab == best)
        fx, fy, fw, fh = _tighten_face_bbox(comp)

    pad_x = int(round(0.10 * fw))
    pad_y = int(round(0.12 * fh))
    fx2 = max(0, fx - pad_x)
    fy2 = max(0, fy - pad_y)
    fx3 = min(w - 1, fx + fw + pad_x)
    fy3 = min(h - 1, fy + fh + pad_y)
    return (fx2, fy2, max(1, fx3 - fx2 + 1), max(1, fy3 - fy2 + 1))


def detect_face_roi(image: np.ndarray, person_mask: np.ndarray,
                    cascade_path: Optional[str] = None,
                    *, info: Optional[Dict[str, Any]] = None
                    ) -> Tuple[int, int, int, int]:
    """返回脸部包围盒 ``(x, y, w, h)``（原图像素）。

    **主路径 = Anime Face Detector（``lbpcascade_animeface``，LBP 级联）**：
    先由级联给出「头部框」，再在框内用肤色连通域收窄到「脸部框」（去头发/脖子）。

    回退链（任一步失败自动降级，不会抛错）：
    级联 + 肤色细化 → 级联原始框 → 全图肤色启发式 → 头部区域中部经验框。

    Parameters
    ----------
    image : np.ndarray  RGB/RGBA 图像
    person_mask : np.ndarray  人物剪影（0/255 或 bool），用于抑制背景误检
    cascade_path : str | None  级联 xml 路径（默认自动探测模块同目录）
    info : dict | None  传入时会被填上检测器诊断信息（实际用的方法、耗时、原始框）
    """
    rgb = _to_rgb(image)
    mask = _to_bool(person_mask)
    h, w = mask.shape

    ann = detect_anime_face(rgb, cascade_path, person_mask=mask)
    detect_ms = float(ann["detect_ms"]) if ann is not None else 0.0
    if info is not None:
        info["detect_ms"] = round(detect_ms, 2)
        info["cascade_path"] = (cascade_path or _default_cascade_path())
        info["cascade_available"] = ann is not None or load_anime_face_cascade(cascade_path) is not None

    if ann is not None:
        bx, by, bw, bh = ann["bbox"]
        if info is not None:
            info.update({
                "name": "lbpcascade_animeface",
                "raw_bbox": [bx, by, bw, bh],
                "n_candidates": ann["n_candidates"],
                "weight": round(float(ann["weight"]), 3),
                "cascade_init_ms": ann["init_ms"],
                "cascade_detect_ms": ann["detect_ms"],
                "search_region": ann["search_region"],
            })
        pad = 0.08
        win = (bx - pad * bw, by - pad * bh, bw * (1 + 2 * pad), bh * (1 + 2 * pad))
        refined = _face_bbox_from_skin(image, person_mask, window=win)
        if refined is not None:
            rx, ry, rw, rh = refined
            # 细化框必须落在头部框附近（防止肤色把肩/臂误当脸）
            cx, cy = rx + rw / 2.0, ry + rh / 2.0
            inside = (bx - 0.15 * bw <= cx <= bx + 1.15 * bw) and (by - 0.15 * bh <= cy <= by + 1.05 * bh)
            if inside and 0.25 * bw <= rw <= 1.15 * bw:
                # 用级联框几何约束脸的上沿：动漫「头部框」的上 30% 是头发，
                # 这样即使肤色分割把发丝算进来（缩放/抗锯齿后常见），也不会把刘海塞进眼睛 ROI。
                rx, ry, rw2, rh2 = refined
                top_min = int(round(by + 0.30 * bh))
                if ry < top_min:
                    rh2 = max(8, rh2 - (top_min - ry))
                    ry = top_min
                if info is not None:
                    info["refined_by_skin"] = True
                    info["face_top_prior"] = top_min
                return (int(rx), int(ry), int(rw2), int(rh2))
        if info is not None:
            info["refined_by_skin"] = False
        return (int(bx), int(by), int(bw), int(bh))

    if info is not None:
        info["name"] = "skin_heuristic"
        info["fallback_reason"] = ("cascade_missing" if load_anime_face_cascade(cascade_path) is None
                                   else "no_face_detected")

    fb = _face_bbox_from_skin(image, person_mask, None)
    if fb is not None:
        return tuple(int(v) for v in fb)

    # 最后兜底：头部区域中部经验框
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, 0, w, h
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    figure_h = max(1, y1 - y0 + 1)
    xmin, xmax, _, _ = _row_extents(mask)
    rows = np.arange(y0, y1 + 1)
    axis = float(np.median((xmin[rows] + xmax[rows]) / 2.0))
    fh = max(16.0, 0.5 * 0.30 * figure_h)
    fw = max(16.0, 0.55 * (x1 - x0 + 1) * 0.5)
    fx = axis - fw / 2.0
    fy = y0 + 0.35 * figure_h - fh / 2.0
    return (int(round(fx)), int(round(fy)), int(round(fw)), int(round(fh)))


# ---------------------------------------------------------------------------
# 4. 眼睛
# ---------------------------------------------------------------------------

def _contour_points(cnt: np.ndarray, max_points: int = 48) -> List[List[int]]:
    """把 OpenCV 轮廓转成 [[x, y], ...]（按周长等距抽样，避免 JSON 过大）。"""
    pts = cnt.reshape(-1, 2).astype(np.float64)
    if pts.shape[0] < 3:
        return []
    peri = cv2.arcLength(cnt.reshape(-1, 1, 2), True)
    approx = cv2.approxPolyDP(cnt.reshape(-1, 1, 2), 0.008 * peri, True).reshape(-1, 2)
    if approx.shape[0] < 3:
        approx = pts
    if approx.shape[0] > max_points:
        k = int(math.ceil(approx.shape[0] / max_points))
        approx = approx[::k]
    return [[int(round(float(x))), int(round(float(y)))] for x, y in approx]


def _empty_eyes() -> Dict[str, Any]:
    return {
        "left": {"contour": [], "center": None},
        "right": {"contour": [], "center": None},
        "midpoint": None,
    }


def _eye_candidates(gray: np.ndarray, thr: float,
                    face_region: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """在脸部灰度图上取暗连通域作为「眼睛候选」（只保留落在脸部凸包内的）。"""
    fh, fw = gray.shape[:2]
    dark = (gray < thr).astype(np.uint8)
    if face_region is not None:
        dark = (dark.astype(bool) & face_region).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, connectivity=8)
    face_area = float(fw * fh)
    out: List[Dict[str, Any]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(cent[i][0]), float(cent[i][1])
        if area < max(12, 0.0008 * face_area) or area > 0.25 * face_area:
            continue
        if bh < 0.06 * fh or bh > 0.60 * fh:
            continue
        if bw < 0.05 * fw or bw > 0.60 * fw:
            continue
        ar = bw / max(1.0, float(bh))
        if ar < 0.40 or ar > 3.5:
            continue
        if cy > 0.65 * fh:                       # 眼睛在脸上半部
            continue
        if bw > 0.75 * fw:                       # 横贯整脸的暗块 = 刘海
            continue
        out.append({"label": i, "area": area, "w": bw, "h": bh,
                    "cx": cx, "cy": cy, "ar": ar})
    return out


def _best_eye_pair(cands: List[Dict[str, Any]], fw: float, fh: float):
    if len(cands) < 2:
        return None, float("inf")
    face_cx = fw / 2.0
    best, best_cost = None, float("inf")
    for a in range(len(cands)):
        for b in range(a + 1, len(cands)):
            ca, cb = cands[a], cands[b]
            if abs(ca["cx"] - cb["cx"]) < 1e-6:
                continue
            l, r = (ca, cb) if ca["cx"] < cb["cx"] else (cb, ca)   # 图像左 / 右
            mid = 0.5 * (l["cx"] + r["cx"])
            sym = abs(mid - face_cx) / fw                        # 关于脸中心对称
            dy = abs(l["cy"] - r["cy"]) / fh                     # 高度一致
            size = abs(math.log(l["area"] / max(1.0, r["area"])))
            gap = abs((r["cx"] - l["cx"]) / fw - 0.45)           # 眼距 ~0.45 脸宽
            cost = 3.0 * sym + 2.0 * dy + 0.8 * size + 1.2 * gap
            cost -= 0.06 * math.log(max(l["area"], r["area"]))
            if cost < best_cost:
                best_cost, best = cost, (l, r)
    return best, best_cost


def detect_eyes(image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """在脸部 ROI 内定位左右眼，返回轮廓点列 + 中心 + 双眼中点。

    步骤：灰度 -> 多档暗域阈值（以 Otsu 为基准上下浮动，自动挑出最佳对称双眼对）
    -> 连通域 -> 按「脸上半部 / 面积 / 长宽比 / 关于脸中心对称 / 眼距」配对
    -> ``findContours`` 取轮廓、质心作中心。

    Returns
    -------
    dict
        ``{'left': {'contour': [[x,y],...], 'center': [x,y]|None},
              'right': {...}, 'midpoint': [x,y]|None}``
        ``left`` / ``right`` 是**人物自身**的左右眼（人物左眼在图像右侧）。
    """
    rgb = _to_rgb(image)
    h, w = rgb.shape[:2]
    fx, fy, fw, fh = [int(v) for v in face_bbox]
    fx, fy = max(0, fx), max(0, fy)
    fw, fh = min(fw, w - fx), min(fh, h - fy)
    if fw < 12 or fh < 12:
        return _empty_eyes()

    roi_rgb = rgb[fy:fy + fh, fx:fx + fw]
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 脸部凸包：暗域只在“脸”的范围内取，避免把鬓发/刘海卷进眼睛轮廓
    skin = _skin_mask(roi_rgb)
    face_region = None
    if int(skin.sum()) > 20:
        pts_xy = np.column_stack(np.nonzero(skin)[::-1]).astype(np.int32)  # (x, y)
        hull = cv2.convexHull(pts_xy.reshape(-1, 1, 2))
        reg = np.zeros_like(skin, np.uint8)
        cv2.fillConvexPoly(reg, hull, 1)
        face_region = reg.astype(bool)

    otsu = _otsu(gray)
    base = float(np.clip(otsu * 0.55, 40.0, 95.0))

    best_pair, best_cost, best_thr = None, float("inf"), base
    best_region = face_region
    # 先带脸部凸包约束（全分辨率下更干净），失败再去掉约束重试（小图凸包不可靠）
    for region_try in ([face_region, None] if face_region is not None else [None]):
        for scale in (1.0, 1.18, 1.40, 0.85, 1.65):
            thr = float(np.clip(base * scale, 25.0, 150.0))
            cands = _eye_candidates(gray, thr, region_try)
            pair, cost = _best_eye_pair(cands, fw, fh)
            if pair is not None and cost < best_cost:
                best_pair, best_cost, best_thr = pair, cost, thr
                best_region = region_try
    if best_pair is None:
        return _empty_eyes()
    face_region = best_region

    # 用最终阈值重新取连通域，保证 label 与 pair 一致
    dark = (gray < best_thr).astype(np.uint8)
    if face_region is not None:
        dark = (dark.astype(bool) & face_region).astype(np.uint8)
    dark = dark.astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(dark, connectivity=8)[:2]

    out = _empty_eyes()
    for side_key, c in (("right", best_pair[0]), ("left", best_pair[1])):   # 图像左 = 人物右
        comp = (lab == c["label"]).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        pts = _contour_points(cnt)
        m = cv2.moments(comp, binaryImage=True)
        if m["m00"] > 0:
            cx = m["m10"] / m["m00"] + fx
            cy = m["m01"] / m["m00"] + fy
        else:
            cx, cy = c["cx"] + fx, c["cy"] + fy
        out[side_key] = {
            "contour": [[p[0] + fx, p[1] + fy] for p in pts],
            "center": [float(cx), float(cy)],
        }
    if out["left"]["center"] is not None and out["right"]["center"] is not None:
        out["midpoint"] = [
            (out["left"]["center"][0] + out["right"]["center"][0]) / 2.0,
            (out["left"]["center"][1] + out["right"]["center"][1]) / 2.0,
        ]
    return out


# ---------------------------------------------------------------------------
# 5. 2D 相似变换
# ---------------------------------------------------------------------------

def fit_similarity_2d(src_xy: np.ndarray, dst_xy: np.ndarray,
                      allow_rotation: bool = False) -> Dict[str, Any]:
    """Umeyama 2D 相似变换：``dst ≈ scale * R(rot) @ src + t``。

    Parameters
    ----------
    src_xy, dst_xy : (N, 2) array_like
        对应点对（同一坐标系，通常 ``src`` = 模板投影点，``dst`` = 图像估计点）。
    allow_rotation : bool
        ``False``（默认）时强制 ``rotation_deg = 0``，只求 scale + 平移；
        ``True`` 时允许平面内旋转（推荐仅在估计稳定时使用）。

    Returns
    -------
    dict
        ``{'scale': float, 'rotation_deg': float, 'translation_px': [tx, ty],
        'rmse_px': float, 'n_points': int}``
        ``scale`` 单位是 **像素/米**。
    """
    src = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    if src.shape != dst.shape:
        raise ValueError("src_xy 与 dst_xy 形状必须一致")
    n = src.shape[0]
    if n < 2:
        raise ValueError("至少需要 2 组对应点")
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = src - mu_s
    D = dst - mu_d

    if allow_rotation:
        cov = (D.T @ S) / n
        U, sv, Vt = np.linalg.svd(cov)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, d]) @ Vt
        var_s = (S ** 2).sum() / n
        scale = float((sv * np.array([1.0, d])).sum() / max(var_s, 1e-12))
        theta = math.atan2(R[1, 0], R[0, 0])
        t = mu_d - scale * (R @ mu_s)
        pred = (scale * (R @ S.T)).T + mu_d
    else:
        # 无旋转：最小二乘 scale + 平移（对两组分量联合求解）
        var_s = (S ** 2).sum() / n
        scale = float((S * D).sum() / max(n * var_s, 1e-12))
        theta = 0.0
        t = mu_d - scale * mu_s
        pred = scale * S + mu_d

    resid = np.linalg.norm(dst - pred, axis=1)
    return {
        "scale": scale,
        "rotation_deg": float(math.degrees(theta)),
        "translation_px": [float(t[0]), float(t[1])],
        "rmse_px": float(np.sqrt((resid ** 2).mean())),
        "n_points": int(n),
    }


def _apply_similarity(P: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """把 2D 点按 ``scale/rotation_deg/translation_px`` 变换到图像坐标。"""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 2)
    s = float(params["scale"])
    th = math.radians(float(params["rotation_deg"]))
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = np.asarray(params["translation_px"], dtype=np.float64)
    return s * (P @ R.T) + t.reshape(1, 2)


# ---------------------------------------------------------------------------
# 6. 关节点估计
# ---------------------------------------------------------------------------

def _analysis_skeleton(image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """从剪影中测量人物几何（头部/肩线/躯干/腿/脚），供关节估计使用。"""
    h, w = mask.shape
    xmin, xmax, counts, ext = _row_extents(mask)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("人物 mask 为空")
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    fig_w = x1 - x0 + 1
    fig_h = y1 - y0 + 1

    rows = np.arange(y0, y1 + 1)
    valid = ext[rows] > 0
    centers = np.where(valid, (xmin[rows] + xmax[rows]) / 2.0, np.nan)
    axis = float(np.nanmedian(centers))

    # ---- 手臂带：整幅横向最宽的连续行段 ----
    wide = (ext > 0.62 * fig_w) & (counts > 0.45 * ext)
    wide[: y0] = False
    wide[y1 + 1:] = False
    yy = np.flatnonzero(wide)
    arm_top = arm_bot = None
    if yy.size:
        peak = int(yy[np.argmax(ext[yy])])
        a = peak
        while a - 1 >= y0 and wide[a - 1]:
            a -= 1
        b = peak
        while b + 1 <= y1 and wide[b + 1]:
            b += 1
        arm_top, arm_bot = int(a), int(b)
    if arm_top is None:                      # 兜底：没有明显横臂
        arm_top = int(y0 + 0.20 * fig_h)
        arm_bot = int(y0 + 0.26 * fig_h)
    arm_thick = max(2, arm_bot - arm_top + 1)
    shoulder_y = arm_top + 0.32 * arm_thick

    # ---- 躯干：手臂带下方第一个明显变窄的行 ----
    torso_top = arm_bot + 1
    torso_row = arm_bot + 1
    for yv in range(arm_bot + 1, y1 + 1):
        if ext[yv] > 0 and ext[yv] < 0.62 * fig_w and counts[yv] > 0.6 * ext[yv]:
            torso_row = int(yv)
            break
    torso_half = float(ext[torso_row]) / 2.0 if ext[torso_row] > 0 else 0.20 * fig_w

    # ---- 裆部（双腿分叉）：连续出现两段“像腿”的 run ----
    def _two_legs(yv: int):
        rr = [(s, e) for s, e in _runs(mask[yv]) if (e - s + 1) > 0.05 * fig_w]
        rr = [r for r in rr if (r[1] - r[0] + 1) < 0.48 * fig_w]
        if len(rr) >= 2:
            rr.sort(key=lambda r: r[0])
            l, r = rr[0], rr[-1]
            gap = r[0] - l[1] - 1
            if gap > max(1.0, 0.008 * fig_w):
                return l, r, gap
        return None

    crotch_y = None
    start = int(arm_bot + 0.15 * fig_h)
    for yv in range(start, y1 - 2):
        got = _two_legs(yv)
        if got and all(_two_legs(min(y1, yv + k)) for k in (6, 12, 18)):
            crotch_y = int(yv)
            break
    if crotch_y is None:
        crotch_y = int(y0 + 0.60 * fig_h)

    legs = _two_legs(crotch_y) or (
        (int(axis - 0.12 * fig_w), int(axis - 0.02 * fig_w), 0),
        (int(axis + 0.02 * fig_w), int(axis + 0.12 * fig_w), 0),
    )
    leg_l, leg_r = legs[0], legs[1]        # 图像左 / 图像右
    leg_l_cx = 0.5 * (leg_l[0] + leg_l[1])
    leg_r_cx = 0.5 * (leg_r[0] + leg_r[1])

    def _leg_center_at(yv: int, side: str) -> float:
        """在某一高度取腿中心线 x（在 crotch 处已知中心附近找 run）。"""
        ref = leg_l_cx if side == "L" else leg_r_cx
        best, bestd = None, 1e9
        for s, e in _runs(mask[yv]):
            if (e - s + 1) < 0.02 * fig_w:
                continue
            c = 0.5 * (s + e)
            d = abs(c - ref)
            if d < bestd:
                bestd, best = d, c
        return float(ref if best is None else best)

    def _leg_width_at(yv: int, side: str) -> float:
        ref = leg_l_cx if side == "L" else leg_r_cx
        best, bestd, bw = None, 1e9, 0.0
        for s, e in _runs(mask[yv]):
            c = 0.5 * (s + e)
            d = abs(c - ref)
            if d < bestd:
                bestd, best, bw = d, c, float(e - s + 1)
        return bw

    # ---- 脚底 / 踝 / 膝 ----
    feet_bottom = y1
    leg_len = max(1.0, float(feet_bottom - crotch_y))

    # 脚底：从下往上找第一段“足够宽”的腿，忽略最底部 2~5 px 的阴影/抗锯齿残影
    scan_y = np.arange(int(crotch_y) + 2, y1 + 1)
    tot_w = np.array([_leg_width_at(int(yv), "L") + _leg_width_at(int(yv), "R")
                      for yv in scan_y], dtype=np.float64)
    soles_y = int(y1)
    if tot_w.size:
        ok = np.flatnonzero(tot_w >= max(6.0, 0.12 * float(tot_w.max())))
        if ok.size:
            soles_y = int(scan_y[ok[-1]])
    stature = max(8.0, float(soles_y - y0))

    # 几何踝：小腿段的最细处（仅作“真实踝不会高于此值”的下界参考）
    ankle_geom = None
    win_lo = int(crotch_y + 0.62 * leg_len)
    win_hi = int(soles_y - 0.05 * leg_len)
    if win_hi - win_lo > 5:
        ys_scan = np.arange(win_lo, win_hi)
        seg = scan_y
        widths = np.interp(ys_scan, seg, tot_w) if tot_w.size else np.zeros(ys_scan.size)
        if widths.size:
            smooth = ndi.uniform_filter1d(widths.astype(np.float64), size=9)
            ankle_geom = int(ys_scan[int(np.argmin(smooth))])
    if ankle_geom is None:
        ankle_geom = int(soles_y - 0.03 * fig_h)

    def _leg_tot_width(yv: int) -> float:
        return _leg_width_at(int(yv), "L") + _leg_width_at(int(yv), "R")

    def _knee_refine(prior: float, ankle: float) -> float:
        """在膝部“先验位置”附近用腿宽局部极小做细化（膝盖上方是最细处，下方是腓肠肌隆起）。"""
        lo = int(max(crotch_y + 4, prior - 0.060 * stature))
        hi = int(min(ankle - 0.10 * (ankle - crotch_y), prior + 0.020 * stature))
        if hi - lo < 12:
            return float(prior)
        ys_k = np.arange(lo, hi + 1)
        wk = np.array([_leg_tot_width(int(yv)) for yv in ys_k], dtype=np.float64)
        if wk.size < 12 or (wk.max() - wk.min()) < 2.0:
            return float(prior)                      # 腿型太“直”，没有可用极小值
        wk_s = ndi.uniform_filter1d(wk, size=9)
        return float(ys_k[int(np.argmin(wk_s))])

    def _foot_center(y_lo: float, y_hi: float, side: str) -> Tuple[float, float]:
        """用脚部区域自身的外形求质心（不假设脚和腿在同一直线上）。"""
        a0 = int(np.clip(np.floor(y_lo), 0, h - 1))
        a1 = int(np.clip(np.ceil(y_hi), 0, h - 1))
        ref = leg_l_cx if side == "L" else leg_r_cx
        pts_x, pts_y = [], []
        for yv in range(a0, a1 + 1):
            best, bd = None, 1e9
            for s, e in _runs(mask[yv]):
                c = 0.5 * (s + e)
                d = abs(c - ref)
                if d < bd:
                    bd, best = d, (s, e)
            if best is not None and bd < 0.15 * fig_w:   # 只取“这一条腿”的那一段
                s, e = best
                pts_y.extend([yv] * (e - s + 1))
                pts_x.extend(range(s, e + 1))
        if not pts_x:
            return float(ref), float(0.5 * (y_lo + y_hi))
        return float(np.mean(pts_x)), float(np.mean(pts_y))

    return {
        "bbox": (x0, y0, x1, y1),
        "fig_w": fig_w, "fig_h": fig_h,
        "axis": axis,
        "arm_top": arm_top, "arm_bot": arm_bot,
        "arm_thickness": arm_thick,
        "shoulder_y": float(shoulder_y),
        "torso_top": torso_top, "torso_row": torso_row,
        "torso_half": torso_half,
        "crotch_y": int(crotch_y),
        "leg_l_cx": float(leg_l_cx), "leg_r_cx": float(leg_r_cx),
        "soles_y": float(soles_y),
        "stature": float(stature),
        "_knee_refine": _knee_refine,
        "_foot_center": _foot_center,
        "ankle_y": float(ankle_geom), "feet_bottom": float(feet_bottom),
        "_leg_center_at": _leg_center_at,
        "_leg_width_at": _leg_width_at,
    }


def _nearest_inside(mask: np.ndarray) -> np.ndarray:
    """返回 (2,H,W) 的最近 mask 像素索引，可把任意点吸附回剪影内（兜底用）。"""
    return ndi.distance_transform_edt(~mask, return_distances=True, return_indices=True)[1]


def _joint(name: str, xy: Optional[Sequence[float]], confidence: float,
           source: str, smplx_index: Optional[int],
           derived_from: Optional[List[str]] = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "name": name,
        "smplx_index": smplx_index,
        "xy": None if xy is None else [float(xy[0]), float(xy[1])],
        "confidence": float(round(confidence, 3)),
        "source": source,
    }
    if derived_from:
        d["derived_from"] = derived_from
    return d


def estimate_joints(image: np.ndarray, person_mask: np.ndarray,
                    J_template: np.ndarray, joint_names: List[str],
                    *, analysis: Optional[Dict[str, Any]] = None,
                    face_bbox: Optional[Tuple[int, int, int, int]] = None,
                    eyes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """估计原图 2D 像素坐标下的 SMPL-X 关节，返回 ``{joint_name: {...}}``。

    每个关节::

        {"name": str, "smplx_index": int|None, "xy": [x, y]|None,
         "confidence": float, "source": "measured|fraction|derived|heuristic"}

    算法（几何先验 + 模板比例）：
      1. 剪影测量：对称轴、头/颈、肩线（手臂带）、躯干宽、裆部、双腿中心线、踝；
      2. 高置信关节由测量直接给出（双眼→head/jaw/neck、肩、髋、骨盆）；
      3. 低置信关节用 **SMPL-X 模板的归一化比例** 沿肢体插值（elbow/wrist/hand/knee/ankle/foot）；
      4. 全部关节做「落回剪影内」的兜底吸附。
    """
    mask = _to_bool(person_mask)
    h, w = mask.shape
    if analysis is None:
        analysis = _analysis_skeleton(image, mask)
    fb = face_bbox
    if fb is None:
        fb = detect_face_roi(image, mask)
    if eyes is None:
        eyes = detect_eyes(image, fb)

    # 模板查找（比例先验，全部用相对量，避免动漫比例与真人差异）
    def T(name: str) -> np.ndarray:
        return _template_joint(J_template, joint_names, name)

    J_hip = 0.5 * (T("left_hip") + T("right_hip"))
    J_pelvis = T("pelvis")
    J_knee = 0.5 * (T("left_knee") + T("right_knee"))
    J_ankle = 0.5 * (T("left_ankle") + T("right_ankle"))
    J_foot = 0.5 * (T("left_foot") + T("right_foot"))
    J_sh = 0.5 * (T("left_shoulder") + T("right_shoulder"))
    J_elbow = 0.5 * (T("left_elbow") + T("right_elbow"))
    J_wrist = 0.5 * (T("left_wrist") + T("right_wrist"))
    J_neck = T("neck")
    J_collar = 0.5 * (T("left_collar") + T("right_collar"))
    J_fingers = np.asarray(J_template, dtype=np.float64)[25:40].mean(axis=0)

    def _f(num: float, den: float, default: float) -> float:
        return float(num / den) if abs(den) > 1e-9 else default

    # 纵向比例（模板 -> 图像），以“髋 -> 脚底”为基准
    f_knee = _f(J_knee[1] - J_hip[1], J_foot[1] - J_hip[1], 0.45)
    f_ankle = _f(J_ankle[1] - J_hip[1], J_foot[1] - J_hip[1], 0.93)
    f_pelvis = _f(J_hip[1] - J_pelvis[1], J_foot[1] - J_hip[1], 0.11)
    # 纵向比例（模板 -> 图像），以 pelvis -> neck 为基准（spine 用）
    f_spine1 = _f(J_pelvis[1] - T("spine1")[1], J_pelvis[1] - J_neck[1], 0.24)
    f_spine2 = _f(J_pelvis[1] - T("spine2")[1], J_pelvis[1] - J_neck[1], 0.53)
    f_spine3 = _f(J_pelvis[1] - T("spine3")[1], J_pelvis[1] - J_neck[1], 0.64)
    # 横向比例（手臂，取人物左臂侧：模板 X 左为正），以 shoulder -> 手指尖 为基准
    shL = T("left_shoulder")
    _Jall = np.asarray(J_template, dtype=np.float64)
    J_tip = _Jall[[27, 30, 33, 36, 39]].mean(axis=0)   # 左手指尖（远端指节均值）
    J_palm = _Jall[25:40].mean(axis=0)                 # 左手手掌中心
    arm_ref = max(1e-6, abs(shL[0] - J_tip[0]))        # shoulder -> 指尖
    f_elbow = _f(abs(shL[0] - T("left_elbow")[0]), arm_ref, 0.39)
    f_wrist = _f(abs(shL[0] - T("left_wrist")[0]), arm_ref, 0.77)
    f_collar = _f(abs(T("left_collar")[0]), abs(shL[0]), 0.27)

    a = analysis
    axis = float(a["axis"])
    fig_h = float(a["fig_h"])
    crotch_y = float(a["crotch_y"])
    feet_y = float(a["feet_bottom"])
    shoulder_y = float(a["shoulder_y"])
    torso_half = float(a["torso_half"])
    arm_top = float(a["arm_top"])

    # ---- 手臂端点 ----
    arm_rows = np.arange(int(a["arm_top"]), int(a["arm_bot"]) + 1)
    arm_rows = arm_rows[(arm_rows >= 0) & (arm_rows < h)]
    if arm_rows.size:
        tip_l = int(min(np.min(np.flatnonzero(mask[y])) for y in arm_rows if mask[y].any()))
        tip_r = int(max(np.max(np.flatnonzero(mask[y])) for y in arm_rows if mask[y].any()))
    else:
        tip_l, tip_r = int(axis - 0.45 * a["fig_w"]), int(axis + 0.45 * a["fig_w"])

    # ---- 脸部锚点 ----
    fx, fy, fw, fh = [float(v) for v in fb]
    face_bottom = fy + fh
    eye_mid = eyes.get("midpoint") if eyes else None
    if eye_mid is None:
        eye_mid = [axis, fy + 0.45 * fh]

    # 头/颈/下颌（模板比例 + 脸部几何混合，动漫头部偏大，置信度给低）
    # SMPL-X 的 head(15) 是颅底/头骨原点，位于双眼稍下方（模板: 眼 - 0.043 m）
    head_y = float(np.clip(eye_mid[1] + 0.50 * (face_bottom - eye_mid[1]),
                           fy + 0.10 * fh, face_bottom + 0.10 * fh))
    neck_y = float(min(arm_top - 0.35 * a["arm_thickness"],
                       face_bottom + 0.45 * max(4.0, arm_top - face_bottom)))
    jaw_y = float(min(face_bottom - 0.12 * fh, neck_y))

    # 肩 / 髋 / 骨盆
    sh_half = torso_half
    hip_half = 0.5 * abs(a["leg_r_cx"] - a["leg_l_cx"])
    hip_y = crotch_y - 0.030 * fig_h
    pelvis_y = hip_y - f_pelvis * max(1.0, feet_y - hip_y)

    # ---- 腿：模板比例（以 hip->脚底 为基准）+ 图像细化 ----
    soles_y = float(a.get("soles_y", feet_y))
    stature = float(a.get("stature", soles_y - a["bbox"][1]))
    leg_span = max(1.0, soles_y - hip_y)

    # 踝：模板比例给先验；几何“腿最细处”只能保证“真踝不高于它”
    #     （穿靴/袜时最细处其实是袜口，会明显偏高，所以取 max）
    ankle_y = hip_y + f_ankle * leg_span
    ankle_y = max(ankle_y, float(a["ankle_y"]))
    ankle_y = min(ankle_y, soles_y - 0.030 * stature)
    ankle_y = max(ankle_y, hip_y + 0.55 * leg_span)      # 别跑到大腿上

    # 膝：模板比例先验 -> 用腿宽局部极小细化（膝盖上方最细，下方是腓肠肌）
    knee_prior = hip_y + f_knee * leg_span
    try:
        knee_y = float(a["_knee_refine"](knee_prior, ankle_y))
    except Exception:
        knee_y = knee_prior
    knee_y = float(np.clip(knee_y, crotch_y + 0.06 * leg_span, ankle_y - 0.18 * leg_span))

    # 脚：用脚部区域自身外形（质心）定位，不假设脚与腿共线
    foot_top = ankle_y
    foot_bot = soles_y
    foot_y = 0.5 * (foot_top + foot_bot)

    def _leg_x(y: float, side: str) -> float:
        yy = int(np.clip(round(y), 0, h - 1))
        return float(a["_leg_center_at"](yy, side))

    # 图像左 = 人物右；图像右 = 人物左（人物左半身用 leg_r_cx 等“图像右”量）

    joints: Dict[str, Any] = {}

    def put(name: str, xy, conf: float, src: str, idx: Optional[int] = None,
            derived_from: Optional[List[str]] = None):
        sidx = idx if idx is not None else SMPLX_INDEX.get(name)
        joints[name] = _joint(name, xy, conf, src, sidx, derived_from)

    # --- 中轴 ---
    put("pelvis", (axis, pelvis_y), 0.75, "measured")
    put("spine1", (axis, pelvis_y - f_spine1 * (pelvis_y - neck_y)), 0.7, "fraction")
    put("spine2", (axis, pelvis_y - f_spine2 * (pelvis_y - neck_y)), 0.7, "fraction")
    put("spine3", (axis, pelvis_y - f_spine3 * (pelvis_y - neck_y)), 0.7, "fraction")
    put("neck", (axis, neck_y), 0.7, "heuristic")
    put("head", (axis, head_y), 0.55, "heuristic")
    put("jaw", (axis + 0.02 * fw, jaw_y), 0.45, "heuristic")

    # --- 左右成对 ---
    for side, sgn, legkey in (("left", +1.0, "R"), ("right", -1.0, "L")):
        # 髋（人物左在图像右，所以 x 大的一侧是 left）
        hip_x = (a["leg_r_cx"] if side == "left" else a["leg_l_cx"])
        put(f"{side}_hip", (hip_x, hip_y), 0.8, "measured")
        put(f"{side}_knee", (_leg_x(knee_y, legkey), knee_y), 0.75, "fraction")
        put(f"{side}_ankle", (_leg_x(ankle_y, legkey), ankle_y), 0.7, "fraction")
        try:
            fcx, fcy = a["_foot_center"](foot_top, foot_bot, legkey)
        except Exception:
            fcx, fcy = _leg_x(foot_y, legkey), foot_y
        put(f"{side}_foot", (fcx, fcy), 0.6, "derived")

        sh_x = axis + sgn * sh_half
        tip = tip_r if side == "left" else tip_l
        L = abs(tip - sh_x)
        elbow_x = sh_x + sgn * f_elbow * L
        wrist_x = sh_x + sgn * f_wrist * L
        # 手掌中心：手腕与指尖的中点附近（不依赖模板绝对位置）
        hand_x = wrist_x + sgn * 0.45 * abs(tip - wrist_x)
        put(f"{side}_shoulder", (sh_x, shoulder_y), 0.85, "measured")
        put(f"{side}_elbow", (elbow_x, shoulder_y), 0.75, "fraction")
        put(f"{side}_wrist", (wrist_x, shoulder_y), 0.75, "fraction")
        put(f"{side}_collar", (axis + sgn * f_collar * sh_half, neck_y + 0.45 * (shoulder_y - neck_y)),
            0.6, "fraction")
        put(f"{side}_hand", (hand_x, shoulder_y), 0.5, "derived", idx=None,
            derived_from=[f"{side}_wrist",
                          "arm_tip (fingertips, joints 25-39)" if side == "left"
                          else "arm_tip (fingertips, joints 40-54)"])

    # --- 眼睛（SMPL-X 23/24 = 眼球中心）---
    if eyes and eyes.get("left", {}).get("center") and eyes.get("right", {}).get("center"):
        put("left_eye_smplhf", eyes["left"]["center"], 0.9, "measured")
        put("right_eye_smplhf", eyes["right"]["center"], 0.9, "measured")
    else:
        put("left_eye_smplhf", (axis + 0.12 * fw, fy + 0.42 * fh), 0.3, "heuristic")
        put("right_eye_smplhf", (axis - 0.12 * fw, fy + 0.42 * fh), 0.3, "heuristic")

    # --- 兜底：把跑出剪影的点吸附回剪影内 ---
    snap_idx = _nearest_inside(mask)
    for name, j in joints.items():
        xy = j.get("xy")
        if xy is None:
            continue
        xi = int(np.clip(round(xy[0]), 0, w - 1))
        yi = int(np.clip(round(xy[1]), 0, h - 1))
        if not mask[yi, xi]:
            sy, sx = snap_idx[0][yi, xi], snap_idx[1][yi, xi]
            j["xy"] = [float(sx), float(sy)]
            j["source"] = j["source"] + "+snapped"

    # 保持输出顺序 = 官方索引顺序 + 派生手
    ordered: Dict[str, Any] = {}
    for _, nm in REQUIRED_JOINTS:
        if nm in joints:
            ordered[nm] = joints[nm]
    for nm in DERIVED_JOINTS:
        if nm in joints:
            ordered[nm] = joints[nm]
    return ordered


# ---------------------------------------------------------------------------
# 7. 可视化
# ---------------------------------------------------------------------------

_PALETTE = {
    "joint": (255, 64, 0),        # 估计骨架：橙红 (RGB)
    "joint_l": (255, 96, 0),
    "joint_r": (0, 140, 255),
    "bone": (255, 64, 0),
    "template": (200, 0, 255),    # 模板投影：洋红
    "face": (0, 170, 80),
    "face_cascade": (0, 200, 255),
    "eye": (0, 230, 120),
    "eye_center": (255, 255, 255),
    "mid": (255, 220, 0),
}


def _halo(rgb: np.ndarray, x: float, y: float) -> Tuple[int, int, int]:
    """依据局部亮度返回描边色（亮背景用黑描边，暗背景用白描边）。"""
    h, w = rgb.shape[:2]
    xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
    r = 6
    patch = rgb[max(0, yi - r): yi + r + 1, max(0, xi - r): xi + r + 1]
    if patch.size == 0:
        return (0, 0, 0)
    lum = float(cv2.cvtColor(patch.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).mean())
    return (0, 0, 0) if lum > 120 else (255, 255, 255)


def _stroke_line(img, p0, p1, color, thick, halo_w=2):
    c0 = (int(round(p0[0])), int(round(p0[1])))
    c1 = (int(round(p1[0])), int(round(p1[1])))
    hc = _halo(img, c0[0], c0[1])
    cv2.line(img, c0, c1, hc, thick + 2 * halo_w, cv2.LINE_AA)
    cv2.line(img, c0, c1, color, thick, cv2.LINE_AA)


def _stroke_dashed(img, p0, p1, color, thick, dash, halo_w=2):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    L = float(np.linalg.norm(p1 - p0))
    if L < 1e-6:
        return
    d = (p1 - p0) / L
    t = 0.0
    while t < L:
        a = p0 + d * t
        b = p0 + d * min(t + dash, L)
        _stroke_line(img, a, b, color, thick, halo_w)
        t += dash * 2.0


def _stroke_circle(img, c, r, color, thick, fill=False, halo_w=2):
    ci = (int(round(c[0])), int(round(c[1])))
    hc = _halo(img, ci[0], ci[1])
    if fill:
        cv2.circle(img, ci, int(r), hc, -1, cv2.LINE_AA)
        cv2.circle(img, ci, int(max(1, r - thick)), color, -1, cv2.LINE_AA)
    else:
        cv2.circle(img, ci, int(r), hc, int(thick + 2 * halo_w), cv2.LINE_AA)
        cv2.circle(img, ci, int(r), color, int(thick), cv2.LINE_AA)


def _stroke_poly(img, pts, color, thick, halo_w=2, closed=True):
    if not pts or len(pts) < 2:
        return
    arr = np.asarray(pts, dtype=np.int32)
    hc = _halo(img, float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    cv2.polylines(img, [arr], closed, hc, int(thick + 2 * halo_w), cv2.LINE_AA)
    cv2.polylines(img, [arr], closed, color, int(thick), cv2.LINE_AA)


def _label(img, text, org, scale, color):
    c = (int(round(org[0])), int(round(org[1])))
    cv2.putText(img, text, c, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 
                max(2, int(scale * 4)), cv2.LINE_AA)
    cv2.putText(img, text, c, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                max(1, int(scale * 2)), cv2.LINE_AA)
    cv2.putText(img, text, c, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                max(1, int(scale)), cv2.LINE_AA)


def visualize(image: np.ndarray, result: Dict[str, Any],
              J_template_projected: Optional[np.ndarray] = None) -> np.ndarray:
    """在图像副本上绘制标注，返回新图像（**不修改输入**）。

    绘制内容：脸部 bbox、双眼轮廓/中心、双眼中点（大十字 + 圆环）、
    估计骨架（橙红，左右区分，带黑/白自适应描边）、模板投影骨架（洋红虚线）、图例。
    线宽与半径随图像尺寸自适应。
    """
    img = _to_rgb(image).copy()
    h, w = img.shape[:2]
    s = float(min(h, w)) / 500.0
    s = max(0.6, min(2.2, s))
    thick = max(1, int(round(2.0 * s)))
    r_joint = max(2.0, 3.6 * s)

    # 1) 模板投影骨架（先画，压在下层）
    if J_template_projected is not None:
        P = np.asarray(J_template_projected, dtype=np.float64)
        if P.ndim == 2 and P.shape[1] >= 2 and P.shape[0] >= 25:
            dash = max(4.0, 7.0 * s)
            for p, c in _BODY_EDGES:
                _stroke_dashed(img, P[p, :2], P[c, :2], _PALETTE["template"],
                               max(1, thick - 1), dash, halo_w=1)
            for i in range(25):
                _stroke_circle(img, P[i, :2], max(1.5, 2.6 * s),
                               _PALETTE["template"], max(1, thick - 1), halo_w=1)

    # 2) 脸部 bbox（+ Anime Face Detector 原始框）
    fbc = result.get("face_bbox_cascade")
    if fbc:
        x, y, fw, fh = [int(v) for v in fbc]
        dash = max(4.0, 8.0 * s)
        for (a, b) in (((x, y), (x + fw, y)), ((x + fw, y), (x + fw, y + fh)),
                       ((x + fw, y + fh), (x, y + fh)), ((x, y + fh), (x, y))):
            _stroke_dashed(img, a, b, _PALETTE["face_cascade"], max(1, thick - 1), dash, halo_w=1)
        _label(img, "anime-face", (x, y - int(6 * s)), 0.40 * s, _PALETTE["face_cascade"])

    fb = result.get("face_bbox")
    if fb:
        x, y, fw, fh = [int(v) for v in fb]
        cv2.rectangle(img, (x, y), (x + fw, y + fh), (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + fw, y + fh), _PALETTE["face"], thick, cv2.LINE_AA)

    # 3) 眼睛
    eyes = result.get("eyes") or {}
    for key in ("left", "right"):
        e = eyes.get(key) or {}
        if e.get("contour"):
            _stroke_poly(img, e["contour"], _PALETTE["eye"], max(1, thick - 1), halo_w=1)
        if e.get("center"):
            _stroke_circle(img, e["center"], max(2.0, 3.0 * s),
                           _PALETTE["eye_center"], max(1, thick), fill=True, halo_w=1)
    mid = result.get("eye_midpoint")
    if mid:
        mx, my = float(mid[0]), float(mid[1])
        R = max(6.0, 10.0 * s)
        _stroke_circle(img, (mx, my), R, _PALETTE["mid"], max(1, thick + 1), halo_w=2)
        _stroke_line(img, (mx - 1.5 * R, my), (mx + 1.5 * R, my), _PALETTE["mid"], max(1, thick), 1)
        _stroke_line(img, (mx, my - 1.5 * R), (mx, my + 1.5 * R), _PALETTE["mid"], max(1, thick), 1)
        _label(img, "eye_mid", (mx + 1.8 * R, my - 0.6 * R), 0.45 * s, _PALETTE["mid"])

    # 4) 估计骨架
    joints = result.get("joints") or {}
    xy: Dict[str, np.ndarray] = {}
    for name, j in joints.items():
        if j.get("xy"):
            xy[name] = np.asarray(j["xy"], dtype=np.float64)
    for p, c in _BODY_EDGES:
        n1, n2 = SMPLX_JOINT_NAMES[p], SMPLX_JOINT_NAMES[c]
        if n1 in xy and n2 in xy:
            col = _PALETTE["joint_l"] if n1.startswith("left") else (
                _PALETTE["joint_r"] if n1.startswith("right") else _PALETTE["bone"])
            _stroke_line(img, xy[n1], xy[n2], col, thick, halo_w=2)
    for n1, n2 in _HAND_EDGES:
        if n1 in xy and n2 in xy:
            _stroke_line(img, xy[n1], xy[n2], _PALETTE["bone"], thick, halo_w=2)
    for name, p in xy.items():
        col = _PALETTE["joint_l"] if name.startswith("left") else (
            _PALETTE["joint_r"] if name.startswith("right") else _PALETTE["joint"])
        if name in DERIVED_JOINTS:
            _stroke_circle(img, p, r_joint * 0.8, col, max(1, thick - 1), halo_w=1)
        else:
            _stroke_circle(img, p, r_joint, col, max(1, thick), fill=True, halo_w=1)

    # 5) 图例
    items = [(_PALETTE["joint"], "estimated skeleton"),
             (_PALETTE["template"], "SMPLX template (fitted)"),
             (_PALETTE["eye"], "eyes / eye-mid"),
             (_PALETTE["face_cascade"], "anime-face detector box")]
    lw, lh = int(232 * s), int(20 * s) * len(items) + int(14 * s)
    # 选一个不遮挡关节/脸框的角落（四角里取与“已绘制内容”重叠面积最小的）
    content = [p for p in xy.values()]
    if result.get("face_bbox"):
        fx0, fy0, fw0, fh0 = [float(v) for v in result["face_bbox"]]
        content += [np.array([fx0, fy0]), np.array([fx0 + fw0, fy0 + fh0])]
    if content:
        P = np.asarray(content)
        bx0, by0 = float(P[:, 0].min()), float(P[:, 1].min())
        bx1, by1 = float(P[:, 0].max()), float(P[:, 1].max())
    else:
        bx0 = by0 = 0.0
        bx1, by1 = float(w), float(h)
    m = int(10 * s)

    def _ov(rect):
        ax0, ay0, ax1, ay1 = rect
        ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0.0, min(ay1, by1) - max(ay0, by0))
        return ix * iy

    cands = [(m, m), (w - lw - m, m), (m, h - lh - m), (w - lw - m, h - lh - m)]
    x0, y0 = min(cands, key=lambda c: _ov((c[0], c[1], c[0] + lw, c[1] + lh)))
    ov = img.copy()
    cv2.rectangle(ov, (x0, y0), (x0 + lw, y0 + lh), (255, 255, 255), -1)
    cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + lw, y0 + lh), (0, 0, 0), 1, cv2.LINE_AA)
    for i, (c, txt) in enumerate(items):
        yy = y0 + int(13 * s) + i * int(19 * s)
        cv2.line(img, (x0 + int(7 * s), yy), (x0 + int(30 * s), yy), c, max(2, thick), cv2.LINE_AA)
        _label(img, txt, (x0 + int(37 * s), yy + int(4 * s)), 0.36 * s, (30, 30, 30))
    return img


# ---------------------------------------------------------------------------
# 8. 一站式入口
# ---------------------------------------------------------------------------

def _read_image(path: str) -> np.ndarray:
    """读图为 RGB 或 RGBA ``uint8``（保留 alpha 供剪影使用）。"""
    with Image.open(path) as im:
        has_alpha = ("A" in im.getbands()) or ("transparency" in im.info)
        im = im.convert("RGBA" if has_alpha else "RGB")
        return np.asarray(im).copy()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    return obj


def estimate(image_path: str,
             smplx_gender: str = "NEUTRAL",
             smplx_dir: str = DEFAULT_SMPLX_DIR,
             output_dir: Optional[str] = None,
             allow_rotation: bool = False,
             write_vis: bool = True,
             face_cascade: Optional[str] = None) -> Dict[str, Any]:
    """一站式入口：估计 2D 关节并写出 ``_annotated.png`` / ``_result.json``。

    Parameters
    ----------
    image_path : str
        输入图（支持 PNG/JPG；透明背景与纯色背景均可）。
    smplx_gender : {"NEUTRAL","MALE","FEMALE"}
    smplx_dir : str
        本地 SMPL-X ``.npz`` 目录（只读）。
    output_dir : str | None
        输出目录；``None`` 表示与输入同目录。
    allow_rotation : bool
        是否允许 ``image_to_model_2d`` 带平面内小旋转。
    write_vis : bool
        ``False`` 时不写 ``_annotated.png``（``--no-vis``）。
    face_cascade : str | None
        Anime Face Detector 级联 xml 路径（``lbpcascade_animeface.xml``）。
        ``None`` 时自动在模块目录 / 当前目录探测；文件不存在则自动回退到肤色启发式。

    Returns
    -------
    dict
        见 README 的字段说明；核心字段：``joints`` / ``eyes`` / ``eye_midpoint`` /
        ``face_bbox`` / ``smplx.image_to_model_2d`` / ``elapsed_ms``。
    """
    t_all0 = time.perf_counter()
    image_path = os.path.abspath(image_path)
    arr = _read_image(image_path)
    rgb = _to_rgb(arr)
    h, w = rgb.shape[:2]

    t0 = time.perf_counter()
    model_path, gender_used = _npz_path_for(smplx_gender, smplx_dir)
    J_template, joint_names, joint_indices = load_smplx_template(gender_used, smplx_dir)
    t_load = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    mask_u8 = detect_person_mask(arr)
    mask = mask_u8 > 0
    t_mask = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    analysis = _analysis_skeleton(rgb, mask)
    t_analysis = (time.perf_counter() - t0) * 1000.0

    face_info: Dict[str, Any] = {}
    t0 = time.perf_counter()
    face_bbox = detect_face_roi(rgb, mask_u8, face_cascade, info=face_info)
    t_face_det = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    eyes = detect_eyes(rgb, face_bbox)
    t_eyes = (time.perf_counter() - t0) * 1000.0
    t_face = t_analysis + t_face_det + t_eyes

    t0 = time.perf_counter()
    joints = estimate_joints(rgb, mask_u8, J_template, joint_names,
                             analysis=analysis, face_bbox=face_bbox, eyes=eyes)
    t_joints = (time.perf_counter() - t0) * 1000.0

    # --- 模板投影 + 2D 相似变换拟合 ---
    t0 = time.perf_counter()
    P2 = np.stack([J_template[:, 0], -J_template[:, 1]], axis=1)   # 正交投影，y 翻转

    fit_pairs: List[Tuple[str, float]] = []
    for idx, nm in REQUIRED_JOINTS:
        j = joints.get(nm)
        if j and j.get("xy"):
            conf = float(j.get("confidence", 0.5))
            fit_pairs.append((nm, conf))
    src = np.array([P2[SMPLX_INDEX[nm]] for nm, _ in fit_pairs], dtype=np.float64)
    dst = np.array([joints[nm]["xy"] for nm, _ in fit_pairs], dtype=np.float64)
    weights = np.array([c for _, c in fit_pairs], dtype=np.float64)

    if len(fit_pairs) >= 2:
        # 加权：把权重折算成重复采样（简单、稳定）
        rep = np.clip(np.round(weights / max(weights.min(), 1e-6)), 1, 4).astype(int)
        src_w = np.repeat(src, rep, axis=0)
        dst_w = np.repeat(dst, rep, axis=0)
        try:
            sim = fit_similarity_2d(src_w, dst_w, allow_rotation=allow_rotation)
        except Exception:
            sim = fit_similarity_2d(src, dst, allow_rotation=allow_rotation)
    else:
        fig_h = float(analysis["fig_h"])
        sim = {"scale": fig_h / 1.6, "rotation_deg": 0.0,
               "translation_px": [float(analysis["axis"]), float(analysis["crotch_y"])],
               "rmse_px": float("nan"), "n_points": 0}

    # 额外给出「核心躯干」拟合：只用最稳定的躯干/头部关节，
    # 在动漫比例与真人差异较大时，比全关节拟合更适合作为编辑器初始对齐。
    CORE = ["pelvis", "left_hip", "right_hip", "left_shoulder", "right_shoulder",
            "neck", "head", "left_eye_smplhf", "right_eye_smplhf"]
    core = [(nm, joints[nm]) for nm in CORE
            if nm in joints and joints[nm].get("xy")]
    sim_core = None
    if len(core) >= 3:
        src_c = np.array([P2[SMPLX_INDEX[nm]] for nm, _ in core], dtype=np.float64)
        dst_c = np.array([j["xy"] for _, j in core], dtype=np.float64)
        try:
            sim_core = fit_similarity_2d(src_c, dst_c, allow_rotation=allow_rotation)
            sim_core["joints_used"] = [nm for nm, _ in core]
        except Exception:
            sim_core = None

    proj = _apply_similarity(P2, sim)

    # 残差（按关节，便于编辑器诊断）
    residuals = {}
    for nm, _ in fit_pairs:
        p = proj[SMPLX_INDEX[nm]]
        residuals[nm] = float(np.linalg.norm(p - np.asarray(joints[nm]["xy"])))
    t_fit = (time.perf_counter() - t0) * 1000.0

    # --- 比例诊断（模板 vs 图像）---
    def _dist(n1: str, n2: str) -> Optional[float]:
        a1, a2 = joints.get(n1), joints.get(n2)
        if not a1 or not a2 or not a1.get("xy") or not a2.get("xy"):
            return None
        return float(np.linalg.norm(np.asarray(a1["xy"]) - np.asarray(a2["xy"])))

    sh_w = _dist("left_shoulder", "right_shoulder")
    hip_w = _dist("left_hip", "right_hip")
    height_px = float(analysis["fig_h"])
    _shl = _template_joint(J_template, joint_names, "left_shoulder")
    _shr = _template_joint(J_template, joint_names, "right_shoulder")
    _hip_l = _template_joint(J_template, joint_names, "left_hip")
    _hip_r = _template_joint(J_template, joint_names, "right_hip")
    _ratio_t = abs(_shl[0] - _shr[0]) / max(1e-9, abs(_hip_l[0] - _hip_r[0]))
    proportion = {
        "shoulder_width_px": sh_w,
        "hip_width_px": hip_w,
        "height_px": height_px,
        "shoulder_to_hip_ratio": (sh_w / hip_w) if (sh_w and hip_w) else None,
        "shoulder_to_hip_ratio_template": float(_ratio_t),
        "height_over_hip_width": (height_px / hip_w) if hip_w else None,
        "armspan_px": _dist("left_hand", "right_hand"),
        "note": "动漫角色头部偏大、手臂偏短，比例与 SMPL-X 真人有系统性差异，"
                "故 image_to_model_2d 的拟合残差偏大属正常现象。",
    }

    elapsed_ms = (time.perf_counter() - t_all0) * 1000.0 - t_load

    result: Dict[str, Any] = {
        "image_path": image_path,
        "image_width": int(w),
        "image_height": int(h),
        "elapsed_ms": int(round(elapsed_ms)),
        "method": "opencv_silhouette_flat_bg_tpose_prior + smplx_template_prior",
        "smplx": {
            "gender": gender_used,
            "model_path": model_path,
            "joint_names": list(joint_names),
            "joint_index_map": {n: i for i, n in enumerate(joint_names)},
            "pixels_per_meter": float(sim["scale"]),
            "image_to_model_2d": {
                "scale": float(sim["scale"]),
                "rotation_deg": float(sim["rotation_deg"]),
                "translation_px": [float(v) for v in sim["translation_px"]],
                "direction": "model_2d -> image_px (orthographic, z dropped, y flipped)",
                "rmse_px": float(sim["rmse_px"]),
                "n_points": int(sim["n_points"]),
                "per_joint_residual_px": residuals,
            },
            "template_projection": {
                "convention": "m = [X, -Y]; p = scale * R(rot) * m + translation_px",
                "J_template_shape": list(np.asarray(J_template).shape),
            },
            "fit_core_torso": sim_core,
            "hand_index_note": "SMPL-X 官方无 hand 关节: 22=jaw, 23/24=双眼, "
                               "25-39/40-54=手指；left_hand/right_hand 为派生手掌中心，"
                               "smplx_index=null。",
        },
        "face_bbox": [int(v) for v in face_bbox],
        "face_bbox_cascade": (list(face_info["raw_bbox"])
                              if face_info.get("raw_bbox") else None),
        "eyes": {
            "left": eyes.get("left"),
            "right": eyes.get("right"),
        },
        "eye_midpoint": eyes.get("midpoint"),
        "joints": joints,
        "conventions": {
            "pixel_origin": "top-left",
            "x_axis": "right",
            "y_axis": "down",
            "units": "original image pixels",
            "left_right": "character-relative (SMPL-X semantics): "
                          "character's LEFT side appears on the image's RIGHT",
            "projection": "orthographic; model (X left, Y up, Z depth) -> m=[X,-Y]; "
                          "p = scale*R(rot)*m + translation_px",
        },
        "diagnostics": {
            "face_detector": face_info,
            "proportion_report": proportion,
            "background": ("alpha" if (np.asarray(arr).ndim == 3 and np.asarray(arr).shape[2] == 4
                                       and float((np.asarray(arr)[..., 3] < 250).mean()) > 0.005)
                           else "flat_color"),
            "timing_ms": {
                "load_template": round(t_load, 1),
                "mask": round(t_mask, 1),
                "face_eyes": round(t_face, 1),
                "analysis": round(t_analysis, 1),
                "face_detect": round(t_face_det, 1),
                "eyes": round(t_eyes, 1),
                "joints": round(t_joints, 1),
                "fit": round(t_fit, 1),
                "total": round((time.perf_counter() - t_all0) * 1000.0, 1),
            },
        },
    }
    # 逐关节残差用于诊断
    result["diagnostics"]["joint_template_residual_px"] = residuals

    out_dir = output_dir or os.path.dirname(image_path)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    if write_vis:
        vis = visualize(rgb, result, J_template_projected=proj)
        vis_path = os.path.join(out_dir, f"{stem}_annotated.png")
        Image.fromarray(vis).save(vis_path)
        result["annotated_path"] = vis_path

    json_path = os.path.join(out_dir, f"{stem}_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(result), f, ensure_ascii=False, indent=2)
    result["result_json_path"] = json_path
    return result
