#!/usr/bin/env python3
"""
手部关键点 3D 可视化脚本，用来检查数据集

从 lerobot keypoints_v5 数据集的 parquet 文件读取 action，提取左右手各 6 个关键点
(wrist, thumb, index, middle, ring, pinky)，在 Open3D 中渲染 3D 坐标和移动轨迹，
并导出 10fps 的 MP4 视频。

用法:
python viz_hand_keypoints_open3d.py \
    --parquet /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_300_keypoints_v5/data/chunk-000/episode_000001.parquet \
    --output /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_300_keypoints_v5/data/hand_keypoints.mp4

"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import open3d as o3d
import open3d.visualization.rendering as rendering

# 每只手 6 个关键点: wrist, thumb, index, middle, ring, pinky
KEYPOINT_NAMES = ["wrist", "thumb", "index", "middle", "ring", "pinky"]
# left: [0:18], right: [21:39], 各 6*3=18 维 xyz
LEFT_XYZ_SLICE = slice(0, 18)
RIGHT_XYZ_SLICE = slice(21, 39)

# 每个关键点一种颜色 (R,G,B) 0-1
KEYPOINT_COLORS = np.array([
    [1.0, 1.0, 1.0],   # wrist - 白色
    [1.0, 0.2, 0.2],   # thumb - 红
    [1.0, 0.6, 0.0],   # index - 橙
    [1.0, 0.9, 0.0],   # middle - 黄
    [0.2, 0.8, 0.2],   # ring - 绿
    [0.2, 0.4, 1.0],   # pinky - 蓝
], dtype=np.float64)


def parse_action_keypoints(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    从 action 数组解析左右手 6 个关键点 xyz。
    action 45 维: left[0:21], right[21:42], waist[42:45]
    每个 hand: wrist[0:3], thumb[3:6], index[6:9], middle[9:12], ring[12:15], pinky[15:18], rotvec[18:21]
    """
    left_xyz = np.asarray(actions[:, LEFT_XYZ_SLICE], dtype=np.float64)   # (T, 18)
    right_xyz = np.asarray(actions[:, RIGHT_XYZ_SLICE], dtype=np.float64)  # (T, 18)
    # reshape to (T, 6, 3)
    left_pts = left_xyz.reshape(-1, 6, 3)
    right_pts = right_xyz.reshape(-1, 6, 3)
    return left_pts, right_pts


def create_keypoint_sphere(radius: float = 0.008):
    """创建用于表示关键点的小球"""
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.compute_vertex_normals()
    return s


def build_trajectory_lineset(
    left_traj: np.ndarray,
    right_traj: np.ndarray,
    max_frame: int,
) -> o3d.geometry.LineSet:
    """
    构建轨迹 LineSet。
    left_traj: (T, 6, 3), right_traj: (T, 6, 3)
    为每个关键点画一条从 frame 0 到 max_frame 的折线。
    """
    n_frames = max_frame + 1
    points = []
    lines = []
    colors = []

    # 左手 6 条轨迹
    for kp_idx in range(6):
        base = len(points)
        for t in range(n_frames):
            points.append(left_traj[t, kp_idx].tolist())
        for t in range(n_frames - 1):
            lines.append([base + t, base + t + 1])
        c = KEYPOINT_COLORS[kp_idx]
        colors.extend([c.tolist()] * (n_frames - 1))

    # 右手 6 条轨迹
    for kp_idx in range(6):
        base = len(points)
        for t in range(n_frames):
            points.append(right_traj[t, kp_idx].tolist())
        for t in range(n_frames - 1):
            lines.append([base + t, base + t + 1])
        c = KEYPOINT_COLORS[kp_idx]
        colors.extend([c.tolist()] * (n_frames - 1))

    ls = o3d.geometry.LineSet()
    pts_arr = np.array(points, dtype=np.float64)
    lines_arr = np.array(lines, dtype=np.int32).reshape(-1, 2) if lines else np.zeros((0, 2), dtype=np.int32)
    colors_arr = np.array(colors, dtype=np.float64).reshape(-1, 3) if colors else np.zeros((0, 3), dtype=np.float64)

    ls.points = o3d.utility.Vector3dVector(pts_arr)
    ls.lines = o3d.utility.Vector2iVector(lines_arr)
    ls.colors = o3d.utility.Vector3dVector(colors_arr)
    return ls


def build_keypoints_pointcloud(
    left_pts: np.ndarray,
    right_pts: np.ndarray,
) -> o3d.geometry.PointCloud:
    """构建当前帧关键点 PointCloud（12 个点）"""
    pts = np.concatenate([left_pts, right_pts], axis=0)  # (12, 3)
    cols = np.tile(KEYPOINT_COLORS, (2, 1))              # (12, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def get_scene_bounds(left_traj: np.ndarray, right_traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算场景包围盒的中心和范围"""
    all_pts = np.concatenate([left_traj.reshape(-1, 3), right_traj.reshape(-1, 3)], axis=0)
    center = np.mean(all_pts, axis=0)
    extents = np.ptp(all_pts, axis=0)
    # 最小范围，避免退化
    extents = np.maximum(extents, 0.3)
    return center, extents


def get_camera_params(
    view: str,
    center: np.ndarray,
    extents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 (eye, up) 用于 setup_camera。
    相机坐标系：X 右，Y 下，Z 前。
    """
    max_ext = float(np.max(extents))
    dist = max_ext * 2.5

    if view == "tilted":
        # 当前视角：从 -Z 略偏右偏上，可见 3D 深度
        eye = center + np.array([0.18 * dist, -0.2 * dist, -0.92 * dist])
        up = np.array([0, -1, 0])
    elif view == "y_neg":
        # 站在 Y 负方向看 Y 正方向（俯视）
        eye = center + np.array([0, -dist, 0])
        up = np.array([0, 0, 1])
    elif view == "z_neg":
        # 站在 Z 负方向看 Z 正方向（正对光学轴）
        eye = center + np.array([0, 0, -dist])
        up = np.array([0, -1, 0])
    else:
        raise ValueError(f"未知视角: {view}")

    return eye, up


def render_frame(
    render: rendering.OffscreenRenderer,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    left_traj: np.ndarray,
    right_traj: np.ndarray,
    frame_idx: int,
    center: np.ndarray,
    extents: np.ndarray,
    view: str = "tilted",
) -> np.ndarray:
    """
    渲染单帧，返回 RGB 图像 (H, W, 3) uint8
    """
    # 移除上一帧的几何体
    for name in ["trajectory", "keypoints"] + [f"kp_{i}" for i in range(12)]:
        try:
            render.scene.remove_geometry(name)
        except Exception:
            pass

    # 轨迹 (从 0 到当前帧)，仅当有至少 2 个点时添加
    if frame_idx >= 1:
        lineset = build_trajectory_lineset(left_traj, right_traj, frame_idx)
        mat_line = rendering.MaterialRecord()
        mat_line.shader = "unlitLine"
        mat_line.line_width = 2.0
        render.scene.add_geometry("trajectory", lineset, mat_line)

    # 当前关键点用小球
    sphere_radius = float(np.mean(extents) * 0.02)
    sphere = create_keypoint_sphere(sphere_radius)
    mat_sphere = rendering.MaterialRecord()
    mat_sphere.shader = "defaultLit"

    pts = np.concatenate([left_pts, right_pts], axis=0)
    for i, pt in enumerate(pts):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        s.compute_vertex_normals()
        s.translate(pt)
        col_idx = i % 6
        c = KEYPOINT_COLORS[col_idx]
        s.paint_uniform_color(c)
        render.scene.add_geometry(f"kp_{i}", s, mat_sphere)

    eye, up = get_camera_params(view, center, extents)
    render.setup_camera(60.0, center, eye, up)
    render.scene.scene.set_sun_light([0, 0, -1], [1.0, 1.0, 1.0], 100000)  # 光从前方(+Z)照来
    render.scene.scene.enable_sun_light(True)
    render.scene.show_axes(True)

    img = render.render_to_image()
    img_np = np.asarray(img)
    # Open3D Image 可能是 UInt8 或 Float
    if img_np.dtype != np.uint8:
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    return img_np


def main():
    parser = argparse.ArgumentParser(description="手部关键点 3D 可视化并导出 MP4")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("/vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_300_keypoints_v5/data/chunk-000/episode_000001.parquet"),
        help="parquet 文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 MP4 路径，默认在 parquet 同目录下",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="输出视频帧率",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="渲染宽度",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="渲染高度",
    )
    args = parser.parse_args()

    parquet_path = args.parquet.resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet 不存在: {parquet_path}")

    if args.output:
        out_dir = Path(args.output).resolve().parent
        out_stem = Path(args.output).stem
    else:
        out_dir = parquet_path.parent
        out_stem = parquet_path.stem

    # 读取 action
    df = pd.read_parquet(parquet_path)
    actions = np.stack(df["action"].to_numpy())
    left_traj, right_traj = parse_action_keypoints(actions)
    n_frames = left_traj.shape[0]

    print(f"加载 parquet: {parquet_path}")
    print(f"  帧数: {n_frames}")
    print(f"  左手关键点形状: {left_traj.shape}")
    print(f"  右手关键点形状: {right_traj.shape}")

    center, extents = get_scene_bounds(left_traj, right_traj)
    print(f"  场景中心: {center}")
    print(f"  场景范围: {extents}")

    # 三种视角：当前倾斜、Y负向俯视、Z负向正视
    VIEWS = [
        ("tilted", "当前倾斜视角"),
        ("y_neg", "Y负向俯视"),
        ("z_neg", "Z负向正视"),
    ]

    render = rendering.OffscreenRenderer(args.width, args.height)
    render.scene.set_background([0.1, 0.1, 0.15, 1.0])

    for view_key, view_desc in VIEWS:
        print(f"\n渲染视角: {view_desc} ({view_key})")
        frames = []
        for t in range(n_frames):
            img = render_frame(
                render,
                left_traj[t],
                right_traj[t],
                left_traj,
                right_traj,
                t,
                center,
                extents,
                view=view_key,
            )
            frames.append(img)
            if (t + 1) % 50 == 0 or t == 0:
                print(f"  渲染帧 {t + 1}/{n_frames}")

        out_path = out_dir / f"{out_stem}_hand_keypoints_{view_key}.mp4"
        saved = False
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
            if out.isOpened():
                for f in frames:
                    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                out.release()
                saved = True
            else:
                raise RuntimeError("cv2.VideoWriter 无法打开")
        except Exception as e:
            print(f"  cv2 导出失败: {e}，尝试 imageio...")
            try:
                import imageio
                imageio.mimsave(str(out_path), frames, fps=args.fps)
                saved = True
            except Exception as e2:
                print(f"  imageio 导出失败: {e2}")
                if view_key == VIEWS[-1][0]:
                    fallback_dir = out_dir / f"{out_stem}_frames"
                    fallback_dir.mkdir(parents=True, exist_ok=True)
                    import imageio
                    for i, f in enumerate(frames):
                        imageio.imwrite(str(fallback_dir / f"frame_{i:04d}.png"), f)
                    print(f"  已保存 PNG 序列到: {fallback_dir}")
                continue

        if saved:
            print(f"  视频已保存: {out_path} ({args.fps} fps)")


if __name__ == "__main__":
    main()
