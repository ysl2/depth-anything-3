#!/usr/bin/env python3
"""
Visualize point cloud using Open3D.
"""
import argparse
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud using Open3D")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to PLY point cloud file")
    parser.add_argument("--width", type=int, default=1920,
                        help="Window width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                        help="Window height (default: 1080)")
    parser.add_argument("--point-size", type=float, default=2.0,
                        help="Point rendering size (default: 2.0)")
    parser.add_argument("--no-rotate", action="store_true",
                        help="Disable X-axis 180° rotation")
    parser.add_argument("--no-center", action="store_true",
                        help="Disable centering")

    args = parser.parse_args()

    # Read point cloud
    print(f"Loading point cloud from {args.input}...")
    pcd = o3d.io.read_point_cloud(args.input)
    print(f"Loaded {len(pcd.points)} points")

    # Get color info
    if pcd.has_colors():
        print("Point cloud has RGB colors")
    else:
        print("Point cloud has no colors, using default coloring")

    # Point cloud bounds
    bounds = pcd.get_axis_aligned_bounding_box()
    min_bound = bounds.min_bound
    max_bound = bounds.max_bound
    print(f"Bounds: X[{min_bound[0]:.2f}, {max_bound[0]:.2f}], "
          f"Y[{min_bound[1]:.2f}, {max_bound[1]:.2f}], "
          f"Z[{min_bound[2]:.2f}, {max_bound[2]:.2f}]")

    # Center the point cloud
    if not args.no_center:
        center = bounds.get_center()
        pcd.translate(-center)
        print(f"Centered point cloud at [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

    # Fix upside-down: rotate 180° around X axis
    if not args.no_rotate:
        R = pcd.get_rotation_matrix_from_xyz([np.pi, 0, 0])
        pcd.rotate(R, center=[0, 0, 0])
        print("Applied X-axis 180° rotation")

    # Set point size using render option
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D Point Cloud Viewer",
                      width=args.width, height=args.height)
    vis.add_geometry(pcd)

    # Get render options and set point size
    opt = vis.get_render_option()
    opt.point_size = args.point_size

    print("Starting visualization...")
    print("Controls:")
    print("  - Mouse drag: Rotate view")
    print("  - Mouse wheel: Zoom")
    print("  - Shift + drag: Pan")
    print("  - Q / ESC: Exit")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
