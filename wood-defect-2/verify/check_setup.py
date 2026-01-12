#!/usr/bin/env python
# check_setup.py - 检查项目环境和数据集

import os
import sys


def check_directory_structure():
    """检查目录结构"""
    print("\n" + "=" * 70)
    print("CHECKING DIRECTORY STRUCTURE")
    print("=" * 70)

    project_root = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_root)

    print(f"\nProject root: {project_root}")
    print(f"Parent directory: {parent_dir}")

    required_dirs = {
        'Dataset (Pine)': '../../pine and rubber dataset/pine dataset',
        'Dataset (Rubber)': '../../pine and rubber dataset/rubber dataset',
        'DINOv2 Model': '../../dinv2-base',
    }

    optional_dirs = {
        'Output': 'wood-defect-output',
        'Checkpoints': 'wood-defect-output/checkpoints',
        'Logs': 'wood-defect-output/logs',
        'Results': 'wood-defect-output/result',
    }

    all_good = True

    print("\nRequired directories (should be in parent directory):")
    for name, path in required_dirs.items():
        full_path = os.path.join(project_root, path)
        full_path = os.path.normpath(full_path)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path}")
        if exists:
            print(f"      → {full_path}")
        if not exists:
            all_good = False

    print("\nOptional directories (will be created automatically in project):")
    for name, path in optional_dirs.items():
        full_path = os.path.join(project_root, path)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "⚠️ (will be created)"
        print(f"  {status} {name}: {path}")

    return all_good


def check_dataset_structure():
    """检查数据集结构"""
    print("\n" + "=" * 70)
    print("CHECKING DATASET STRUCTURE")
    print("=" * 70)

    project_root = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(project_root)

    datasets = {
        'Pine Wood': '../pine and rubber dataset/pine dataset',
        'Rubber Wood': '../pine and rubber dataset/rubber dataset'
    }

    required_subdirs = ['JPEGImages', 'SegmentationClass', 'ImageSets/Segmentation']
    required_files = ['ImageSets/Segmentation/train.txt',
                      'ImageSets/Segmentation/val.txt']

    all_good = True

    for dataset_name, dataset_path in datasets.items():
        print(f"\n{dataset_name}:")
        full_dataset_path = os.path.join(project_root, dataset_path)
        full_dataset_path = os.path.normpath(full_dataset_path)

        if not os.path.exists(full_dataset_path):
            print(f"  ❌ Dataset not found: {dataset_path}")
            print(f"     Expected at: {full_dataset_path}")
            all_good = False
            continue

        print(f"  ✅ Found at: {full_dataset_path}")

        # 检查子目录
        for subdir in required_subdirs:
            subdir_path = os.path.join(full_dataset_path, subdir)
            exists = os.path.exists(subdir_path)
            status = "✅" if exists else "❌"
            print(f"  {status} {subdir}/")
            if not exists:
                all_good = False

        # 检查文件
        for file in required_files:
            file_path = os.path.join(full_dataset_path, file)
            exists = os.path.exists(file_path)
            status = "✅" if exists else "❌"
            print(f"  {status} {file}")
            if not exists:
                all_good = False

        # 统计图片数量
        images_dir = os.path.join(full_dataset_path, 'JPEGImages')
        labels_dir = os.path.join(full_dataset_path, 'SegmentationClass')

        if os.path.exists(images_dir):
            num_images = len([f for f in os.listdir(images_dir)
                              if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  📊 Number of images: {num_images}")

        if os.path.exists(labels_dir):
            num_labels = len([f for f in os.listdir(labels_dir)
                              if f.endswith('.png')])
            print(f"  📊 Number of labels: {num_labels}")

    return all_good


def check_python_packages():
    """检查Python包"""
    print("\n" + "=" * 70)
    print("CHECKING PYTHON PACKAGES")
    print("=" * 70)

    required_packages = [
        'torch',
        'torchvision',
        'transformers',
        'numpy',
        'opencv-python',
        'pillow',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'pandas',
        'tqdm',
        'tensorboard'
    ]

    all_good = True

    for package in required_packages:
        try:
            if package == 'opencv-python':
                __import__('cv2')
            elif package == 'pillow':
                __import__('PIL')
            elif package == 'scikit-learn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            all_good = False

    return all_good


def check_gpu():
    """检查GPU"""
    print("\n" + "=" * 70)
    print("CHECKING GPU")
    print("=" * 70)

    try:
        import torch

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"  ✅ CUDA available: {num_gpus} GPU(s)")

            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"     GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print(f"  ⚠️  CUDA not available - will use CPU")
            return False
    except Exception as e:
        print(f"  ❌ Error checking GPU: {e}")
        return False

    return True


def print_usage_instructions():
    """打印使用说明"""
    print("\n" + "=" * 70)
    print("USAGE INSTRUCTIONS")
    print("=" * 70)

    print("\n1. Train the model:")
    print("   python train.py")

    print("\n2. Test the model:")
    print("   python test.py --checkpoint wood-defect-output/checkpoints/best_model.pth")

    print("\n3. Generate visualizations:")
    print("   python visualize.py --checkpoint wood-defect-output/checkpoints/best_model.pth")

    print("\n4. Generate Table 2 results:")
    print("   python evaluate_table2.py --checkpoint wood-defect-output/checkpoints/best_model.pth")

    print("\n5. Quick start (all steps):")
    print("   bash quick_start.sh")


def main():
    print("=" * 70)
    print("LAM WOOD DEFECT DETECTION - SETUP CHECKER")
    print("=" * 70)

    # 检查各项
    dir_ok = check_directory_structure()
    dataset_ok = check_dataset_structure()
    packages_ok = check_python_packages()
    gpu_ok = check_gpu()

    # 总结
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_ok = dir_ok and dataset_ok and packages_ok

    if all_ok:
        print("\n✅ All checks passed! You're ready to go!")
        print_usage_instructions()
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")

        if not dir_ok or not dataset_ok:
            print("\n📝 To fix dataset issues:")
            print("   1. Create workspace directory structure:")
            print("      workspace/")
            print("      ├── pine and rubber dataset/")
            print("      ├── dinv2-base/")
            print("      └── wood-defect/  (your project)")
            print("")
            print("   2. Move datasets to parent directory:")
            print("      cd ..")
            print("      mv /path/to/datasets/* ./")
            print("")
            print("   3. Move DINOv2 model to parent directory:")
            print("      cd ..")
            print("      mv /path/to/dinv2-base ./")

        if not packages_ok:
            print("\n📝 To install missing packages:")
            print("   pip install torch torchvision transformers")
            print("   pip install opencv-python pillow matplotlib seaborn")
            print("   pip install scikit-learn pandas tqdm tensorboard")

        if not gpu_ok:
            print("\n📝 GPU not available:")
            print("   - Training will be slower on CPU")
            print("   - Modify device in configs/lam_config.py to 'cpu'")


if __name__ == "__main__":
    main()