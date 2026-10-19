# AnimePoseDetector

**动漫人物 T-Pose 轻量 2D 姿态估计**（SMPL-X 语义对齐），供后续整合进 SMPL-X 编辑器。

- 纯 CPU、**无神经网络**；唯一外部权重是 241 KB 的 Anime Face Detector LBP 级联（MIT）
- 单张 768×1152 图 **≈ 107 ms**（稳态，含图像解码）
- 输出**原图像素坐标**下的 SMPL-X 前 25 个关节 + 派生手掌中心 + 双眼轮廓/中心/中点
- 脸部/眼睛：**Anime Face Detector（`lbpcascade_animeface`）** 定位头部 → 肤色细化到脸 → ROI 内暗域 + 对称配对定眼
- 用本地 `SMPLX_*.npz` 的 `v_template` + `J_regressor` 做**几何先验**、**骨架叠加**与 **2D 相似变换**
- 支持 **透明背景**、**纯白背景**、**任意纯色背景**

---

## 1. 环境与依赖

| 依赖 | 版本（本机实测） | 说明 |
|---|---|---|
| Python | 3.10.20 | conda 环境 `torch3d` |
| numpy | 2.2.6 | 未改动 |
| scipy | 1.15.3 | 已存在，用于 `binary_fill_holes` / `distance_transform_edt` |
| pillow | 12.2.0 | 已存在，读写图片 |
| opencv-python-headless | 4.12.0.88 | **新增**（54 MB wheel，Apache-2.0，`pip install --no-deps`，不影响 numpy/torch） |
| `lbpcascade_animeface.xml` | 246,945 B (241 KB) | **新增**（项目目录内的模型文件，**MIT**，纯 OpenCV LBP 级联，无需任何 Python 新依赖） |
| torch / pytorch3d | 2.14.0.dev20260809+cu130 / 0.7.9 | **未被使用、未改动** |

> 本模块 **不 import torch**，因此对 PyTorch3D 环境零影响，也省掉约 1 s 的 torch 导入开销。

安装（若环境里还没有 cv2）：

```bash
conda run -n torch3d pip install --no-deps "opencv-python-headless==4.12.0.88"
```

Anime Face Detector 级联（已放在项目目录，若缺失可重新获取）：

```bash
curl -fsSL -o lbpcascade_animeface.xml \
  https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml
```

- 大小 **241 KB**，远小于 150 MB 限制
- 许可证 **MIT**（版权声明内嵌在 xml 文件头部：`Copyright (C) 2011 by nagadomi@nurs.or.jp`）
- 由 `cv2.CascadeClassifier` 直接加载，**不引入任何新的 Python 依赖**，也不碰 torch/numpy
- 找不到文件时自动回退到肤色启发式（结果不变，只是少一层模型先验）

---

## 2. 快速开始

```bash
conda run -n torch3d python /home/lomerson/MyProjects/AnimePoseDetector/main.py \
  --input  /home/lomerson/MyProjects/AnimePoseDetector/ref.png \
  --smplx-dir /home/lomerson/MyProjects/models/smplx \
  --smplx-gender NEUTRAL
```

输出（默认与输入同目录）：

- `<stem>_annotated.png` —— 标注图
- `<stem>_result.json` —— 完整结果

可选参数：

| 参数 | 含义 |
|---|---|
| `--output-dir DIR` | 指定输出目录 |
| `--no-vis` | 不写标注图（只出 JSON） |
| `--allow-rotation` | 允许 `image_to_model_2d` 带平面内小旋转 |
| `--face-cascade PATH` | 指定 Anime Face Detector 级联 xml（默认自动探测 `./lbpcascade_animeface.xml`） |

Python 内使用（编辑器集成）：

```python
from anime_pose_detector import estimate, load_smplx_template, fit_similarity_2d

res = estimate("ref.png")                       # 一站式
res["joints"]["left_wrist"]["xy"]               # [x, y] 原图像素
res["smplx"]["image_to_model_2d"]               # scale / rotation_deg / translation_px
res["eye_midpoint"]                             # 双眼中点
```

---

## 3. 坐标与投影约定（`image_to_model_2d`）

SMPL-X 模板坐标：`X` = 人物左右（**左为正**）、`Y` = 向上、`Z` = 前后（前为正，T-Pose 下 `|Z| < 0.13 m`）。

采用**正交投影并丢弃 Z**：

```
m   = [X, -Y]                                   # 米；y 翻转为图像方向（向下）
p   = scale * R(rotation_deg) @ m + t           # 像素
```

| 字段 | 单位 | 含义 |
|---|---|---|
| `scale` | 像素/米 | 即 `pixels_per_meter` |
| `rotation_deg` | 度 | 图像平面内旋转，按图像坐标系（x 右 / y 下）逆时针为正；`allow_rotation=False` 时恒为 0 |
| `translation_px` | 像素 | 模型原点（`(0,0,0)`，即 pelvis 高度处的中轴点）在图像中的位置 |

方向是 **model → image**（把模板骨架投到图上），字段名沿用任务约定 `image_to_model_2d`。
逆变换：`m = R(-rot) @ (p - t) / scale`。

**`fit_core_torso`**：额外给出的“只用躯干/头部 9 个关节”的拟合结果（`pelvis`、双髋、双肩、`neck`、`head`、双眼）。
动漫角色头大臂短，与 SMPL-X 真人比例存在系统性差异，全关节拟合会被手臂/腿拉偏；
躯干拟合更稳，**建议编辑器初始化优先用它**，再让用户微调。

---

## 4. 输出 JSON 字段

```jsonc
{
  "image_path": "/abs/path/ref.png",
  "image_width": 768,
  "image_height": 1152,
  "elapsed_ms": 166,                     // 稳态整条 pipeline（不含模板加载）
  "method": "opencv_silhouette_flat_bg_tpose_prior + smplx_template_prior",

  "smplx": {
    "gender": "NEUTRAL",
    "model_path": ".../SMPLX_NEUTRAL.npz",
    "joint_names": [ ... 55 个官方关节名 ... ],
    "joint_index_map": { "pelvis": 0, "left_hip": 1, ... },   // 名字 -> 官方索引
    "pixels_per_meter": 548.84,
    "image_to_model_2d": {
      "scale": 548.84,
      "rotation_deg": 0.0,
      "translation_px": [380.7, 346.4],
      "direction": "model_2d -> image_px (orthographic, z dropped, y flipped)",
      "rmse_px": 36.4,
      "n_points": 44,
      "per_joint_residual_px": { "pelvis": 9.1, ... }
    },
    "fit_core_torso": {                     // 建议的编辑器初始对齐
      "scale": 565.44, "rotation_deg": 0.0,
      "translation_px": [379.7, 353.4],
      "rmse_px": 16.93, "n_points": 9,
      "joints_used": ["pelvis", "left_hip", ...]
    },
    "template_projection": { "convention": "m = [X, -Y]; p = scale*R(rot)*m + t",
                             "J_template_shape": [55, 3] },
    "hand_index_note": "..."
  },

  "face_bbox": [331, 178, 99, 67],          // 细化后的脸部框 [x, y, w, h]，原图像素
  "face_bbox_cascade": [307, 132, 145, 145],// Anime Face Detector 原始「头部框」（可能为 null）
  "eyes": {
    "left":  { "contour": [[x,y], ...], "center": [409.0, 189.2] },
    "right": { "contour": [[x,y], ...], "center": [352.8, 188.2] }
  },
  "eye_midpoint": [380.8, 189.4],

  "joints": {
    "pelvis": { "name": "pelvis", "smplx_index": 0, "xy": [380.5, 548.5],
                "confidence": 0.75, "source": "measured" },
    "...": " ... 共 25 个官方关节 + 2 个派生手掌中心 ..."
  },

  "conventions": { "pixel_origin": "top-left", "x_axis": "right", "y_axis": "down",
                   "left_right": "character-relative ...", "projection": "..." },

  "diagnostics": {
    "face_detector": {                      // 脸部检测实际用了哪条路径
      "name": "lbpcascade_animeface",       // 或 "skin_heuristic"
      "raw_bbox": [307, 132, 145, 145],
      "n_candidates": 1, "weight": 5.752,
      "cascade_init_ms": 10.85,             // 级联加载+预热（每进程一次）
      "cascade_detect_ms": 5.69,            // 单次检测
      "search_region": [0, 66, 768, 558],   // 头部区域裁剪
      "refined_by_skin": true,
      "face_top_prior": 175
    },
    "proportion_report": { "shoulder_width_px": 213.0, "hip_width_px": 95.0,
                           "shoulder_to_hip_ratio": 2.24, "height_over_hip_width": 10.92, ... },
    "background": "flat_color",             // 或 "alpha"
    "timing_ms": { "mask": 51.1, "analysis": 5.2, "face_detect": 7.5,
                   "eyes": 2.0, "joints": 22.5, "fit": 0.3, "total": 106.7 },
    "joint_template_residual_px": { ... }
  }
}
```

### 每个关节的字段

| 字段 | 含义 |
|---|---|
| `name` | 关节名（见第 5 节） |
| `smplx_index` | SMPL-X 官方索引；派生关节（`left_hand`/`right_hand`）为 `null` |
| `xy` | `[x, y]` 原图像素（左上原点、x 右、y 下）；无法估计时为 `null` |
| `confidence` | 0–1 置信度（`measured` 直接测量 > `fraction` 按模板比例插值 > `heuristic`） |
| `source` | `measured` / `fraction` / `heuristic` / `derived`，可能带 `+snapped` 后缀（曾吸附回剪影内） |
| `derived_from` | 仅派生关节：说明由哪些量推导 |

---

## 5. 关节索引表（SMPL-X 官方 55）

| idx | name | idx | name | idx | name |
|---|---|---|---|---|---|
| 0 | pelvis | 19 | right_elbow | 38 | left_thumb2 |
| 1 | left_hip | 20 | left_wrist | 39 | left_thumb3 |
| 2 | right_hip | 21 | right_wrist | 40 | right_index1 |
| 3 | spine1 | 22 | **jaw** | 41 | right_index2 |
| 4 | left_knee | 23 | **left_eye_smplhf** | 42 | right_index3 |
| 5 | right_knee | 24 | **right_eye_smplhf** | 43 | right_middle1 |
| 6 | spine2 | 25 | left_index1 | 44 | right_middle2 |
| 7 | left_ankle | 26 | left_index2 | 45 | right_middle3 |
| 8 | right_ankle | 27 | left_index3 | 46 | right_pinky1 |
| 9 | spine3 | 28 | left_middle1 | 47 | right_pinky2 |
| 10 | left_foot | 29 | left_middle2 | 48 | right_pinky3 |
| 11 | right_foot | 30 | left_middle3 | 49 | right_ring1 |
| 12 | neck | 31 | left_pinky1 | 50 | right_ring2 |
| 13 | left_collar | 32 | left_pinky2 | 51 | right_ring3 |
| 14 | right_collar | 33 | left_pinky3 | 52 | right_thumb1 |
| 15 | head | 34 | left_ring1 | 53 | right_thumb2 |
| 16 | left_shoulder | 35 | left_ring2 | 54 | right_thumb3 |
| 17 | right_shoulder | 36 | left_ring3 | | |
| 18 | left_elbow | 37 | left_thumb1 | | |

> **注意**：SMPL-X **没有** `left_hand` / `right_hand` 关节（22 = `jaw`，23/24 = 双眼，25–54 = 手指）。
> 本工具仍按任务要求输出 `left_hand` / `right_hand`（手掌中心），但 `smplx_index = null`，
> 由「手腕 → 指尖」几何推导，不与官方索引冲突。

---

## 6. 左右约定

SMPL-X 中 `X+` 是人物**自身**的左侧；对正面朝向镜头的角色，人物的左手出现在**图像右侧**。
本工具所有 `left_*` / `right_*` 关节、以及 `eyes.left` / `eyes.right` **一律是人物自身左右**：

```
left_shoulder.xy[0] > right_shoulder.xy[0]      # 人物左肩在图像右侧
eyes.left.center[0] > eyes.right.center[0]
```

`eye_midpoint` 不受影响。若你的下游需要“图像左/右”，交换 `left/right` 即可。

---

## 7. 算法

```
                 ┌── alpha 通道可用？ ── 是 ──► mask = alpha > 128
 图像 ──► 剪影 ──┤
                 └── 否 ──► 外圈中位数估背景色 bg，噪声自适应阈值分割
                              └─► 形态学清理 → 填洞 → 最大连通域（耐浅色衣服/皮肤）

 剪影 ──► 逐行扫描：对称轴、手臂带（整幅最宽行段）、躯干宽、裆部（双腿分叉）、
          双腿中心线、脚底（忽略底部 2~5 px 阴影/抗锯齿残影）

 头部区域 ──► Anime Face Detector（LBP 级联，只在剪影上部 52% 内检测）
              └─► 头部框 face_bbox_cascade
                   └─► 框内肤色连通域 → 收窄到下巴 → 脸部框 face_bbox

 脸部框 ──► 凸包内暗域 → 多档阈值 → 对称双眼配对 → findContours → 中心/中点

 关节 ──► 高置信：肩/髋/骨盆/颈/头/双眼  ← 直接测量
          低置信：肘/腕/膝/踝/脚/锁骨/脊柱 ← SMPL-X 模板的**归一化比例**沿肢体插值
          兜底：任何跑出剪影的关节吸附回最近剪影像素

 模板 ──► J_template = J_regressor @ v_template (55×3)
          P2 = [X, -Y]  ──Umeyama──►  image_to_model_2d (scale, rot, t)
          叠加投影骨架（洋红虚线）
```

模板只提供**相对比例**（肘/腕/膝在肢体上的位置、锁骨横向比例、脊柱纵向比例），
不直接套用模板的绝对坐标 —— 动漫角色头大臂短，绝对坐标会整体偏移。

### 腿部：膝 / 踝 / 脚

腿部三个关节都不靠“最细/最宽处”硬认，因为穿靴/袜/裤时最细处会跑偏：

| 关节 | 规则 |
|---|---|
| `ankle` | 以 `hip → 脚底` 为基准，取模板比例 0.931（人腿胫骨占 hip→踝 的 ~93%）；再与“腿最细处”取 **max** —— 真踝**不可能高于**腿最细处，而穿靴时最细处只是袜口，会明显偏高，所以宁可取模板值 |
| `knee` | 模板比例先验（0.451 of hip→脚底）→ 在 `prior ±(0.06H, 0.02H)` 窗口内找**腿宽局部极小**做细化（膝上方最细、下方是腓肠肌隆起）；腿型太“直”时退回先验 |
| `foot` | 用 `[ankle, 脚底]` 这段**脚部自身轮廓的质心**定位，不再假设脚与腿共线；脚趾方向/透视造成的横向偏移会被质心自然吸收 |

锚点用 `hip` 而不是 `crotch`：裆部比股骨头低约 3% 身高，用它当基准会把膝/踝整体压低。

### 脸部检测：Anime Face Detector

| 项 | 值 |
|---|---|
| 模型 | `lbpcascade_animeface.xml`（LBP 级联，OpenCV `CascadeClassifier`） |
| 来源 | https://github.com/nagadomi/lbpcascade_animeface |
| 大小 | 246,945 B（241 KB） |
| 许可证 | **MIT**（声明内嵌于 xml 头部，`Copyright (C) 2011 by nagadomi@nurs.or.jp`） |
| 依赖 | 无（`cv2` 即可，不需要 onnxruntime/torch） |
| 参数 | `scaleFactor=1.1`, `minNeighbors=5`, `minSize=(24,24)`, 灰度 `equalizeHist` |

流程：剪影上部 52% 作为搜索区（比全图快 ~2.5×）→ 级联出「头部框」→ 框内做肤色连通域，
用逐行宽度收到「脸部框」（去头发/脖子）→ 若级联未命中，则回退到全图肤色启发式。

**为什么要有它**：环境自带的 OpenCV 人类脸级联在这类动漫图上**完全检测不到脸**
（实测 `haarcascade_frontalface_default/alt/alt2` 全部 0 命中），而动漫专用级联能稳定命中。

**速度（CPU，实测）**：

| 输入分辨率 | 级联加载+预热 | 检测（头部区域裁剪） | 检测（全图，无 mask 时） |
|---|---|---|---|
| 384×576 | 9.3 ms（每进程一次） | **1.8 ms** | 4.8 ms |
| 768×1152 | 0.1 ms（缓存命中） | **6.1 ms** | 17.0 ms |
| 1152×1728 | 0.1 ms | **13.8 ms** | 30.2 ms |
| 1536×2304 | 0.1 ms | **23.0 ms** | 45.1 ms |

分辨率鲁棒性：同一张图缩放到 0.5×/0.75×/1×/1.5×/2×，眼睛中点折算回原尺度后
落在 `(380.3–382.0, 188.6–189.7)`，**跨尺度偏差 < 2 px**。

### 背景适配

| 情况 | 处理 |
|---|---|
| 透明 PNG（RGBA/LA，alpha 有透明像素） | 直接用 alpha > 128 作剪影 |
| 纯白背景 | 外圈中位数 = 白，噪声自适应阈值（通常 8） |
| 任意纯色背景（绿幕、深灰、浅灰…） | 同上，`||p - bg||` 距离图 + 外圈 MAD 自适应阈值 |
| 渐变 / 复杂背景 | 不保证（会退化为 Otsu，可能失败） |

---

## 8. 性能（本机实测，768×1152）

| 阶段 | 稳态耗时 |
|---|---|
| 模板 npz 加载 + `J_regressor @ v_template` | **3.1 ms**（每进程一次，之后 lru_cache） |
| 图像解码（PIL） | ~13 ms |
| 剪影 + 填洞 + 连通域 | ~51 ms |
| 骨架几何测量 | ~5 ms |
| **脸部检测（级联 + 肤色细化）** | **~7.5 ms** |
| **眼睛（ROI 内）** | **~2.0 ms** |
| 关节 + 吸附 | ~22 ms |
| 相似变换拟合 | ~0.3 ms |
| **稳态总计** | **≈ 107 ms**（5 次：102.4–110.0，中位 106.5） |
| 进程内首次调用 | ≈ 163 ms（多出级联预热 ~11 ms + 模板加载 ~3 ms） |

对比：改用 Anime Face Detector 之前（纯肤色启发式全图搜索）稳态约 **160 ms**，
现在 ≈ 107 ms —— 更快且更稳。CPU 单线程，无需 GPU。

## 9. 已知限制

1. **输入端假设**：单张、正面、左右大致对称、T-Pose、纯色/透明背景。侧身、弯曲手臂、复杂背景不保证。
2. **动漫比例 vs 真人模板**：`ref.png` 的角色头高约占身高 18%（真人 ~13%）、臂展/身高 0.72（真人 ~1.0），
   因此全关节 `image_to_model_2d` 的 `rmse` 偏大（本例 36 px）属正常现象；
   想稳定对齐请用 `fit_core_torso`（本例 rmse 17 px）。
3. `head` / `jaw` / `neck` 是几何启发式（动漫常常“没有脖子”），置信度 0.45–0.7；
   `head(15)` 按 SMPL-X 语义取**颅底**（双眼稍下方），而不是头顶。
4. 手掌中心是派生量（SMPL-X 无该关节），不做手指。
5. 眼睛轮廓会包含睫毛/眼线的暗部，中心取暗块质心，通常与瞳孔接近但不完全相同。
6. Anime Face Detector 给的是「头部框」（含头发），必须再经肤色细化才是「脸部框」；
   极小图（<400 px 高）上细化可能失败，此时 `face_bbox` 直接等于级联框。
7. 长靴/长袜盖住脚踝时，踝关节只能靠模板比例给出（腿最细处会被袜口劫持），
   误差约 1~2% 身高；若你的角色是靴子控，建议在编辑器里手动微调 `left/right_ankle`。
8. `left/right_foot` 取脚部轮廓质心，正面站姿下几乎与踝同 x；侧身/外八时会自然偏向脚尖一侧。
6. 只支持单人物；多人同图时会取最大连通域。

---

## 10. 只读约束

- `models/smplx` 全程**只读**：只 `np.load` 读取 `v_template` 与 `J_regressor`；
  不写、不拷贝、不生成缓存文件，也不读取 `.pkl`（因此不需要 chumpy）。
- 不写任何全局缓存目录。

---

## 11. 文件

| 文件 | 说明 |
|---|---|
| `anime_pose_detector.py` | 核心库（8 个对外函数 + 内部测量/绘制工具） |
| `main.py` | CLI |
| `ref.png` | 示例输入 |
| `ref_annotated.png` / `ref_result.json` | 示例输出 |
