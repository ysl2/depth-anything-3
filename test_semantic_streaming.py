#!/usr/bin/env python3
"""
快速测试脚本 - 使用少量图像测试DA3语义分割
"""
import os
import sys

# 添加src目录到路径
sys.path.insert(0, 'src')
sys.path.insert(0, 'da3_streaming')

from loop_utils.config_utils import load_config
from da3_semantic_streaming import DA3_Semantic_Streaming

def main():
    # 测试配置
    image_dir = "/home/songliyu/Templates/south-building/images"
    config_path = "./da3_streaming/configs/base_config_semantic_16gb.yaml"
    output_dir = "./test_output"

    # 检查图像
    import glob
    images = sorted(
        glob.glob(os.path.join(image_dir, "*.JPG"))
        + glob.glob(os.path.join(image_dir, "*.jpg"))
    )

    if len(images) == 0:
        print(f"No images found in {image_dir}")
        return

    # 只使用前16张图像进行快速测试
    test_images = images[:16]
    print(f"Found {len(images)} images, using {len(test_images)} for testing")

    # 创建临时测试目录（符号链接到测试图像）
    import tempfile
    test_image_dir = tempfile.mkdtemp(prefix="da3_test_")

    for i, img_path in enumerate(test_images):
        os.symlink(img_path, os.path.join(test_image_dir, f"img_{i:04d}.JPG"))

    print(f"Created test directory with {len(test_images)} images: {test_image_dir}")

    # 加载配置
    config = load_config(config_path)

    # 调整chunk_size以适应小数据集
    config["Model"]["chunk_size"] = 8
    config["Model"]["overlap"] = 2
    config["Model"]["loop_enable"] = False  # 禁用回路闭合以加快测试
    config["Model"]["semantic_enable"] = True  # 启用语义分割

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Starting DA3-Semantic-Streaming Test")
    print("=" * 60)

    try:
        # 运行语义增强的DA3-Streaming
        da3_streaming = DA3_Semantic_Streaming(test_image_dir, output_dir, config)
        da3_streaming.run()
        da3_streaming.close()

        print("=" * 60)
        print("Test completed successfully!")
        print(f"Output saved to: {output_dir}")
        print("=" * 60)

        # 列出输出文件
        import os
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"{subindent}{file} ({file_size/1024/1024:.2f} MB)")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理临时目录
        import shutil
        shutil.rmtree(test_image_dir)
        print(f"Cleaned up test directory: {test_image_dir}")

if __name__ == "__main__":
    main()
