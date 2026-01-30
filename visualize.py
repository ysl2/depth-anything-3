import argparse
import open3d as o3d
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud using Open3D")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to PLY point cloud file")
    parser.add_argument("--rotate", action="store_true", default=True,
                        help="Enable X-axis 180° rotation")

    args = parser.parse_args()

    # Read point cloud
    pcd = o3d.io.read_point_cloud(args.input)
    print(f"Loaded {len(pcd.points)} points")

    # Optional rotation
    if args.rotate:
        R = pcd.get_rotation_matrix_from_xyz([np.pi, 0, 0])
        pcd.rotate(R, center=[0, 0, 0])

    # Visualize
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
