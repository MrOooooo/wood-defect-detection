# check_windows_setup.py
"""
Windows环境配置检查脚本
检查数据集和模型路径是否正确
"""

import os
import sys
import torch
from pathlib import Path

# 设置Windows下的路径
DATASET_ROOT = r'/home/user4/桌面/wood-defect/pine and rubber dataset'
DINOV2_MODEL_PATH = r'/home/user4/桌面/wood-defect/dinv2-base'


def check_path_exists(path, description):
    """检查路径是否存在"""
    path_obj = Path(path)
    exists = path_obj.exists()

    status = "✓" if exists else "✗"
    print(f"{status} {description}")
    print(f"  Path: {path}")

    if exists:
        if path_obj.is_dir():
            # 列出目录内容
            items = list(path_obj.iterdir())
            print(f"  Items: {len(items)} items")
            if len(items) <= 10:
                for item in items:
                    print(f"    - {item.name}")
        else:
            # 文件大小
            size = path_obj.stat().st_size / (1024 * 1024)  # MB
            print(f"  Size: {size:.2f} MB")
    else:
        print(f"  ⚠ WARNING: Path does not exist!")

    print()
    return exists


def check_dataset_structure(dataset_root, dataset_name):
    """检查数据集结构"""
    print(f"\n{'=' * 60}")
    print(f"Checking {dataset_name} Dataset Structure")
    print(f"{'=' * 60}\n")

    dataset_path = Path(dataset_root) / f"{dataset_name} dataset"

    if not dataset_path.exists():
        print(f"✗ Dataset not found: {dataset_path}")
        return False

    print(f"✓ Dataset root: {dataset_path}\n")

    # 检查必需的目录
    required_dirs = [
        'ImageSets/Segmentation',
        'JPEGImages',
        'SegmentationClass'
    ]

    all_exist = True
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        exists = dir_path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {dir_name}: {dir_path}")

        if exists and dir_path.is_dir():
            items = list(dir_path.iterdir())
            print(f"  └─ {len(items)} items")

        all_exist = all_exist and exists

    # 检查split文件
    print("\nSplit files:")
    split_dir = dataset_path / 'ImageSets' / 'Segmentation',
    if isinstance(split_dir, tuple):
        # 如果是元组，取第一个元素作为路径
        split_dir = Path(split_dir[0])

    if split_dir.exists():
        for split in ['train.txt', 'val.txt', 'test.txt']:
            split_file = split_dir / split
            if split_file.exists():
                with open(split_file, 'r') as f:
                    num_lines = len([l for l in f.readlines() if l.strip()])
                print(f"  ✓ {split}: {num_lines} images")
            else:
                print(f"  ✗ {split}: Not found")

    # 检查class_names.txt
    print("\nClass names:")
    class_file = dataset_path / 'class_names.txt'
    if class_file.exists():
        with open(class_file, 'r') as f:
            classes = [l.strip() for l in f.readlines() if l.strip()]
        print(f"  ✓ class_names.txt: {len(classes)} classes")
        for i, cls in enumerate(classes):
            print(f"    {i}: {cls}")
    else:
        print(f"  ✗ class_names.txt: Not found")
        print(f"  ⚠ You need to create this file!")

    return all_exist


def check_dinov2_model(model_path):
    """检查DINOv2模型"""
    print(f"\n{'=' * 60}")
    print(f"Checking DINOv2 Model")
    print(f"{'=' * 60}\n")

    model_path_obj = Path(model_path)

    if not model_path_obj.exists():
        print(f"✗ Model path not found: {model_path}")
        print(f"\n⚠ Please download DINOv2 model:")
        print(f"   1. Visit: https://huggingface.co/facebook/dinov2-base")
        print(f"   2. Download all files to: {model_path}")
        print(f"   3. Or use git: git clone https://huggingface.co/facebook/dinov2-base {model_path}")
        return False

    print(f"✓ Model path exists: {model_path}\n")

    # 检查必需的文件
    required_files = [
        'configs.json',
        'model.safetensors',  # 或 pytorch_model.bin
        'preprocessor_config.json'
    ]

    print("Checking model files:")
    all_exist = True
    for file_name in required_files:
        file_path = model_path_obj / file_name
        # 对于模型权重，检查两种可能的文件
        if file_name == 'model.safetensors':
            safetensors_exists = file_path.exists()
            pytorch_bin = model_path_obj / 'pytorch_model.bin'
            pytorch_exists = pytorch_bin.exists()

            if safetensors_exists:
                print(f"  ✓ model.safetensors")
                size = file_path.stat().st_size / (1024 * 1024)
                print(f"    Size: {size:.2f} MB")
            elif pytorch_exists:
                print(f"  ✓ pytorch_model.bin")
                size = pytorch_bin.stat().st_size / (1024 * 1024)
                print(f"    Size: {size:.2f} MB")
            else:
                print(f"  ✗ Model weights not found (model.safetensors or pytorch_model.bin)")
                all_exist = False
        else:
            exists = file_path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {file_name}")
            all_exist = all_exist and exists

    return all_exist


def check_python_packages():
    """检查Python包"""
    print(f"\n{'=' * 60}")
    print(f"Checking Python Packages")
    print(f"{'=' * 60}\n")

    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'transformers': 'Transformers (for DINOv2)',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm'
    }

    for package, description in required_packages.items():
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {description}")
        except ImportError:
            print(f"✗ {description} - NOT INSTALLED")
            print(f"  Install with: pip install {package}")


def check_cuda():
    """检查CUDA"""
    print(f"\n{'=' * 60}")
    print(f"Checking CUDA")
    print(f"{'=' * 60}\n")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")
    else:
        print("⚠ CUDA not available - will use CPU (very slow)")


def generate_config_template():
    """生成配置文件模板"""
    print(f"\n{'=' * 60}")
    print(f"Generating Configuration Template")
    print(f"{'=' * 60}\n")

    config_content = f'''# configs/lam_config.py
"""
LAM模型配置文件 - Windows环境
"""

import os

class Config:
    # ========== 数据集配置 ==========
    dataset_root = r'{DATASET_ROOT}'
    pine_wood_path = os.path.join(dataset_root, 'pine dataset')
    rubber_wood_path = os.path.join(dataset_root, 'rubber dataset')

    # DINOv2模型路径
    dinov2_model_path = r'{DINOV2_MODEL_PATH}'

    # ========== 训练配置 ==========
    batch_size = 4
    num_workers = 4
    num_epochs_pretrain = 10
    num_epochs_full = 20

    # ========== 优化器配置 ==========
    learning_rate = 1e-4
    weight_decay = 0.05

    # ========== 图像配置 ==========
    image_size = 512

    # ========== 模型配置 ==========
    backbone = 'dinov2'
    pine_num_classes = 4
    rubber_num_classes = 6
    num_classes = pine_num_classes

    # ========== LAM模块配置 ==========
    feature_dim = 768  # DINOv2-base
    num_tokens = 100
    token_rank = 16
    num_groups = 16
    lambda_cov = 0.5
    tau = 0.5

    # ========== 保存配置 ==========
    checkpoint_dir = './checkpoints'
    log_dir = './logs'

    # ========== GPU配置 ==========
    device = 'cuda'
    multi_gpu = False
    gpu_ids = [0]

configs = Config()
'''

    # 保存到文件
    config_file = Path('../configs') / 'lam_config_windows.py'
    config_file.parent.mkdir(exist_ok=True)

    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"✓ Configuration template saved to: {config_file}")
    print(f"\nYou can copy this to configs/lam_config.py")


def main():
    print("\n" + "=" * 70)
    print("LAM Model - Windows Environment Setup Check")
    print("=" * 70)

    # 1. 检查数据集路径
    print("\n1. Checking Dataset Root")
    print("-" * 60)
    check_path_exists(DATASET_ROOT, "Dataset Root")

    # 2. 检查Pine Wood数据集
    check_dataset_structure(DATASET_ROOT, 'pine')

    # 3. 检查Rubber Wood数据集
    check_dataset_structure(DATASET_ROOT, 'rubber')

    # 4. 检查DINOv2模型
    check_dinov2_model(DINOV2_MODEL_PATH)

    # 5. 检查Python包
    check_python_packages()

    # 6. 检查CUDA
    check_cuda()

    # 7. 生成配置模板
    generate_config_template()

    # 总结
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nNext steps:")
    print("1. If any paths are missing (✗), please fix them")
    print("2. Copy configs/lam_config_windows.py to configs/lam_config.py")
    print("3. Run: python verify_dataset.py --dataset pine --visualize")
    print("4. If everything looks good, start training: python train.py")
    print("\nFor detailed instructions, see: VOC_QUICKSTART.md")
    print("=" * 70)


if __name__ == "__main__":
    main()