import pyvista as pv
import argparse
import os
import numpy as np
from vtkmodules.vtkIOPLY import vtkPLYWriter

class AxisRotator:
    def __init__(self, mesh, plotter, filename, rgb_name, raw_rgb_data):
        self.mesh = mesh
        self.plotter = plotter
        self.filename = filename
        self.rgb_name = rgb_name
        # 始终保存原始的 0-255 数据 (uint8)
        self.raw_rgb_data = raw_rgb_data

        # 备份原始几何坐标
        self.original_points = mesh.points.copy()

        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0

    def update_mesh(self):
        # 1. 还原坐标
        self.mesh.points = self.original_points.copy()

        # 2. 应用旋转
        if self.angle_x != 0:
            self.mesh.rotate_x(self.angle_x, inplace=True)
        if self.angle_y != 0:
            self.mesh.rotate_y(self.angle_y, inplace=True)
        if self.angle_z != 0:
            self.mesh.rotate_z(self.angle_z, inplace=True)

        self.plotter.render()

    def callback_x(self, value):
        self.angle_x = value
        self.update_mesh()

    def callback_y(self, value):
        self.angle_y = value
        self.update_mesh()

    def callback_z(self, value):
        self.angle_z = value
        self.update_mesh()

    def save_cloud(self, state):
        if not state: return

        base, ext = os.path.splitext(self.filename)
        out_name = f"{base}_corrected{ext}"

        print(f"\n正在保存到: {out_name} ...")

        # --- 关键修改区域：保存前的数据处理 ---
        temp_visual_rgb = None

        if self.rgb_name is not None and self.raw_rgb_data is not None:
            # 1. 暂存当前用于显示的浮点颜色
            temp_visual_rgb = self.mesh[self.rgb_name].copy()

            # 2. 【关键修复】将 RGB 数据恢复为原始 uint8 格式 (0-255)
            # VTK PLYWriter 需要 uint8 类型的 RGB 数据
            self.mesh.point_data[self.rgb_name] = self.raw_rgb_data
            self.mesh.set_active_scalars(self.rgb_name)

        try:
            # 3. 使用 VTK PLYWriter 保存（这是唯一能正确保存 RGB 的方法）
            writer = vtkPLYWriter()
            writer.SetFileName(out_name)
            writer.SetInputData(self.mesh)
            if self.rgb_name is not None:
                writer.SetArrayName(self.rgb_name)
            writer.SetFileType(1)  # 1 = binary format
            writer.Write()

            print(f"成功！已保存为兼容格式")
            self.plotter.add_text("Saved Successfully!", position='upper_left', color='green', font_size=18)
        except Exception as e:
            print(f"保存失败: {e}")
            self.plotter.add_text("Error Saving!", position='upper_left', color='red', font_size=18)

        # --- 恢复显示 ---
        # 保存完后，切回 float 颜色以保持显示正常
        if self.rgb_name is not None and temp_visual_rgb is not None:
            self.mesh.point_data[self.rgb_name] = temp_visual_rgb
            # 恢复显示的活跃状态
            self.mesh.set_active_scalars(self.rgb_name)

def main():
    parser = argparse.ArgumentParser(description="交互式点云轴向校准工具 (Final Fix)")
    parser.add_argument("-i", "--input", required=True, help="Path to input PLY file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        return

    print(f"Loading {args.input}...")
    mesh = pv.read(args.input)

    # ---------------- RGB 数据处理逻辑 ----------------
    rgb_name = None
    raw_rgb_data = None
    rgb_opts = {}

    # 1. 探测颜色数组名称
    if 'RGB' in mesh.array_names:
        rgb_name = 'RGB'
    elif 'rgb' in mesh.array_names:
        rgb_name = 'rgb'

    if rgb_name:
        print(f"Detected color array: {rgb_name}")

        # 2. 备份原始数据 (保留原始 uint8)
        raw_rgb_data = mesh[rgb_name].copy()

        # 3. 准备显示数据
        if raw_rgb_data.max() > 1.0:
            print("Normalizing colors for editor visualization only...")
            # 修改内存中的数据用于显示
            mesh[rgb_name] = raw_rgb_data / 255.0

        # 4. 设置显示参数
        rgb_opts = {
            'scalars': rgb_name,
            'rgb': True,
            'show_scalar_bar': False
        }
    else:
        print("No RGB data found. Using default color.")
        rgb_opts = {'color': 'white'}
    # ------------------------------------------------

    # 中心化
    center = mesh.center
    mesh.points -= center
    print("Cloud centered at (0,0,0) for rotation.")

    # 创建绘图器
    pl = pv.Plotter(window_size=[1600, 1000])
    pl.set_background("white")

    # 添加网格 (匹配你的脚本风格)
    pl.add_mesh(
        mesh,
        point_size=3.0,
        render_points_as_spheres=True,
        lighting=False,
        **rgb_opts
    )

    pl.show_axes()
    pl.show_grid(color='black')
    pl.add_text("Adjust Sliders -> Click [Save]", position='upper_right', color='black', font_size=12)

    # 初始化 Rotator
    rotator = AxisRotator(mesh, pl, args.input, rgb_name, raw_rgb_data)

    # UI 控件
    slider_config = {'style': 'modern', 'color': 'black', 'pointa': (0.05, 0.1), 'pointb': (0.25, 0.1)}
    pl.add_slider_widget(rotator.callback_x, rng=[-180, 180], value=0, title="Rotate X", **slider_config)

    slider_config['pointa'] = (0.35, 0.1)
    slider_config['pointb'] = (0.55, 0.1)
    pl.add_slider_widget(rotator.callback_y, rng=[-180, 180], value=0, title="Rotate Y", **slider_config)

    slider_config['pointa'] = (0.65, 0.1)
    slider_config['pointb'] = (0.85, 0.1)
    pl.add_slider_widget(rotator.callback_z, rng=[-180, 180], value=0, title="Rotate Z", **slider_config)

    # 保存按钮
    pl.add_checkbox_button_widget(
        rotator.save_cloud,
        value=False,
        position=(20, 100),
        size=40,
        border_size=3,
        color_on='green',
        color_off='lightgrey'
    )
    pl.add_text("Save", position=(75, 110), font_size=12, color='black')

    pl.show()

if __name__ == "__main__":
    main()
