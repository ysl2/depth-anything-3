#!/usr/bin/env python3
"""
点云视频录制脚本
围绕点云旋转相机并录制视频
"""
import argparse
import pyvista as pv
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="录制点云旋转视频")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="点云 PLY 文件路径")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="输出视频文件路径 (默认: <input>_video.mp4)")
    parser.add_argument("--fps", type=int, default=30,
                        help="视频帧率 (默认: 30)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="视频时长（秒）(默认: 10.0)")
    parser.add_argument("--azimuth-start", type=float, default=0,
                        help="起始方位角 (默认: 0)")
    parser.add_argument("--elevation", type=float, default=30,
                        help="仰角 (默认: 30)")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="缩放比例 (默认: 1.0)")
    parser.add_argument("--width", type=int, default=1920,
                        help="视频宽度 (默认: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                        help="视频高度 (默认: 1080)")
    parser.add_argument("--point-size", type=float, default=3.0,
                        help="点大小 (默认: 3.0)")
    parser.add_argument("--background", type=str, default="white",
                        help="背景颜色 (默认: white)")
    parser.add_argument("--showcase", action="store_true",
                        help="展示模式：隐藏坐标轴并放大建筑物")
    parser.add_argument("--showcase-zoom", type=float, default=0.5,
                        help="展示模式缩放因子，越小放大越多 (默认: 0.5)")

    args = parser.parse_args()

    # 设置输出路径
    if args.output is None:
        base, _ = args.input.rsplit('.', 1)
        args.output = f"{base}_video.mp4"

    # 读取点云
    print(f"加载点云: {args.input}")
    mesh = pv.read(args.input)
    print(f"点数: {mesh.n_points}, 数组: {mesh.array_names}")

    # 检查 RGB
    has_rgb = 'RGB' in mesh.array_names or 'rgb' in mesh.array_names
    if has_rgb:
        rgb_name = 'RGB' if 'RGB' in mesh.array_names else 'rgb'
        rgb = mesh[rgb_name]
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
            mesh[rgb_name] = rgb
        print(f"使用颜色数组: {rgb_name}")

    # 计算总旋转角度
    total_frames = int(args.fps * args.duration)
    azimuth_per_frame = 360 / total_frames

    print(f"录制参数:")
    print(f"  分辨率: {args.width}x{args.height}")
    print(f"  帧率: {args.fps} FPS")
    print(f"  时长: {args.duration} 秒 ({total_frames} 帧)")
    print(f"  输出: {args.output}")

    # 创建绘图器
    plotter = pv.Plotter(window_size=[args.width, args.height], off_screen=True)
    plotter.set_background(args.background)

    # 添加网格
    if has_rgb:
        plotter.add_mesh(
            mesh,
            scalars=rgb_name,
            rgb=True,
            point_size=args.point_size,
            render_points_as_spheres=True,
            lighting=False,
            show_scalar_bar=False
        )
    else:
        plotter.add_mesh(
            mesh,
            color='white',
            point_size=args.point_size,
            render_points_as_spheres=True,
            lighting=True
        )

    # 自动设置视野
    plotter.reset_camera()

    # 获取初始相机参数，用于计算轨道
    focal_point = mesh.center
    initial_position = np.array(plotter.camera.position)
    initial_distance = np.linalg.norm(initial_position - focal_point)

    # 计算相机轨道半径（根据仰角调整）
    elevation_rad = np.radians(args.elevation)
    orbit_radius = initial_distance / np.cos(elevation_rad)

    # 打印调试信息
    print(f"\n点云中心: {focal_point}")
    print(f"相机距离: {initial_distance:.2f}")
    print(f"轨道半径: {orbit_radius:.2f}")

    # 展示模式：缩小轨道半径以放大建筑物
    if args.showcase:
        orbit_radius = orbit_radius * args.showcase_zoom
        print(f"展示模式：轨道半径调整为 {orbit_radius:.2f}")

    # 显示轴和网格（展示模式下不显示）
    if not args.showcase:
        plotter.show_axes()
        plotter.show_grid(color='black' if args.background == 'white' else 'white')

    # 录制视频
    print("\n开始录制...")
    plotter.open_movie(args.output, framerate=args.fps)

    for i in range(total_frames):
        if i % 30 == 0 or i == total_frames - 1:
            print(f"进度: {i+1}/{total_frames} 帧 ({(i+1)/total_frames*100:.1f}%)")

        # 计算当前轨道位置（水平旋转）
        current_azimuth = np.radians(args.azimuth_start + i * azimuth_per_frame)

        # 球坐标转笛卡尔坐标：相机围绕点云做轨道运动
        cam_x = focal_point[0] + orbit_radius * np.cos(current_azimuth) * np.cos(elevation_rad)
        cam_y = focal_point[1] + orbit_radius * np.sin(current_azimuth) * np.cos(elevation_rad)
        cam_z = focal_point[2] + orbit_radius * np.sin(elevation_rad)

        plotter.camera.position = (cam_x, cam_y, cam_z)
        plotter.camera.focal_point = focal_point
        plotter.camera.up = (0, 0, 1)  # Z轴向上

        plotter.write_frame()

    plotter.close()
    print(f"\n视频已保存到: {args.output}")


if __name__ == "__main__":
    main()
