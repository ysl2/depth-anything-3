import argparse
import numpy as np
import pyvista as pv


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud using PyVista")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to PLY point cloud file")
    parser.add_argument("-w", "--window-name", type=str, default="Point Cloud Viewer",
                        help="Window title")
    parser.add_argument("--width", type=int, default=1920,
                        help="Window width")
    parser.add_argument("--height", type=int, default=1080,
                        help="Window height")
    parser.add_argument("--no-rotate", action="store_true",
                        help="Disable X-axis 180° rotation")
    parser.add_argument("--center", action="store_true",
                        help="Center the point cloud")
    parser.add_argument("--point-size", type=float, default=2.0,
                        help="Point rendering size (default: 2.0)")
    parser.add_argument("--mesh", action="store_true",
                        help="Reconstruct surface mesh (using Poisson reconstruction)")
    parser.add_argument("--opacity", type=float, default=1.0,
                        help="Point opacity 0-1 (default: 1.0)")
    parser.add_argument("--background", type=str, default="white",
                        help="Background color (default: white)")

    args = parser.parse_args()

    # Read point cloud
    print(f"Loading point cloud from {args.input}...")
    mesh = pv.read(args.input)
    print(f"Loaded {mesh.n_points} points, arrays: {mesh.array_names}")

    # Check for RGB colors
    has_rgb = 'RGB' in mesh.array_names or 'rgb' in mesh.array_names
    if has_rgb:
        # Normalize RGB to 0-1 range if needed
        rgb_name = 'RGB' if 'RGB' in mesh.array_names else 'rgb'
        rgb = mesh[rgb_name]
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
            mesh[rgb_name] = rgb
        print(f"Using {rgb_name} colors")

    # Optional centering
    if args.center:
        bounds = mesh.bounds
        center = [(bounds[0] + bounds[1]) / 2,
                  (bounds[2] + bounds[3]) / 2,
                  (bounds[4] + bounds[5]) / 2]
        mesh.translate([-center[0], -center[1], -center[2]], inplace=True)

    # Optional rotation
    if not args.no_rotate:
        mesh.rotate_x(180, point=mesh.center, inplace=True)

    # Surface reconstruction
    if args.mesh:
        print("Reconstructing surface mesh...")
        try:
            # Poisson surface reconstruction
            surf = mesh.reconstruct_surface(progress_bar=True)
            mesh = surf
            print("Surface reconstruction complete")
        except Exception as e:
            print(f"Surface reconstruction failed: {e}")
            print("Falling back to point cloud")

    # Create plotter
    plotter = pv.Plotter()
    plotter.title = args.window_name

    # Add mesh to scene
    if has_rgb and not args.mesh:
        # Use RGB colors directly
        plotter.add_mesh(
            mesh,
            scalars=rgb_name,
            rgb=True,
            render_points_as_spheres=True,
            point_size=args.point_size,
            opacity=args.opacity,
            show_edges=False,
            lighting=False,
            interpolate_before_map=False
        )
    else:
        # Default rendering
        plotter.add_mesh(
            mesh,
            render_points_as_spheres=True,
            point_size=args.point_size,
            opacity=args.opacity,
            show_edges=False,
            lighting=True
        )

    # Set background
    plotter.set_background(args.background)

    # Set window size
    plotter.window_size = (args.width, args.height)

    # Show
    print("Starting visualization...")
    plotter.show()


if __name__ == "__main__":
    main()
