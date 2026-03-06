# README.localhost

1. 可用结果路径：

   ```text
   /home/songliyu/Documents/Depth-Anything-3/da3_streaming/exps/_home_songliyu_Templates_DJI-Mini3-Pro_20260208_102MEDIA_DJI_0544_images/2026-02-08-12-23-07/pcd
   ```

1. 修复轴向：

   ```text
   python3 fix_axis.py -i /home/songliyu/Documents/Depth-Anything-3/da3_streaming/exps/_home_songliyu_Templates_DJI-Mini3-Pro_20260208_102MEDIA_DJI_0544_images/2026-02-08-12-23-07/pcd/combined_pcd.ply
   ```

1. 三维结果可视化：

   ```text
   # 轴向修复前
   python3 visualize_pyvista.py -i /home/songliyu/Documents/Depth-Anything-3/da3_streaming/exps/_home_songliyu_Templates_DJI-Mini3-Pro_20260208_102MEDIA_DJI_0544_images/2026-02-08-12-23-07/pcd/combined_pcd.ply --point-size 3
   # 轴向修复后
   python3 visualize_pyvista.py -i /home/songliyu/Documents/Depth-Anything-3/da3_streaming/exps/_home_songliyu_Templates_DJI-Mini3-Pro_20260208_102MEDIA_DJI_0544_images/2026-02-08-12-23-07/pcd/combined_pcd_corrected.ply
   ```

1. 视频录制：

   ```text
   python3 record_ply_video.py -i /home/songliyu/Documents/Depth-Anything-3/da3_streaming/exps/_home_songliyu_Templates_DJI-Mini3-Pro_20260208_102MEDIA_DJI_0544_images/2026-02-08-12-23-07/pcd/combined_pcd_corrected.ply --showcase --showcase-zoom 0.4
   ```
