# configs/lam_config.py
import os


class Config:
    # ========== 项目根目录 ==========
    # 获取项目根目录(假设config文件在 configs/ 文件夹中)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ========== 上级目录 ==========
    # 数据集和模型放在项目的上级目录
    PARENT_DIR = os.path.dirname(PROJECT_ROOT)

    # ========== 数据集配置 ==========
    # 相对于上级目录的路径
    dataset_root = os.path.join(PARENT_DIR, 'pine and rubber dataset')
    pine_wood_path = os.path.join(dataset_root, 'pine dataset')
    rubber_wood_path = os.path.join(dataset_root, 'rubber dataset')
    dinov2_model_path = os.path.join(PARENT_DIR, 'dinv2-base')

    # ========== 数据集配置 ==========
    # dataset_root = r'/home/user4/桌面/wood-defect/pine and rubber dataset'
    # pine_wood_path = os.path.join(dataset_root, 'pine dataset')
    # rubber_wood_path = os.path.join(dataset_root, 'rubber dataset')
    # dinov2_model_path = r'/home/user4/桌面/wood-defect/dinv2-base'

    # ========== 训练配置(论文设置) ==========
    batch_size = 4  # 论文使用batch_size=4
    num_workers = 4
    num_epochs_pretrain = 10  # 论文:预训练10 epochs
    num_epochs_full = 20      # 论文:完整训练20 epochs
    # patience = 15

    # ========== 优化器配置(论文设置) ==========
    learning_rate = 1e-4  # 论文: 1.0 × 10^-4
    learning_rate_stage2 = 5e-5  # 从1e-4 * 0.5 降到5e-5
    weight_decay = 0.05   # 论文: 0.05
    eps = 1e-8            # 论文: 1.0 × 10^-8
    use_strong_augmentation = True  # 新增配置项

    # ========== 学习率调度器 ==========
    lr_scheduler = 'poly'
    poly_power = 0.9
    min_lr = 0  # 论文中衰减到0

    # ========== 图像配置(论文设置) ==========
    image_size = 512
    crop_range = [256, 1024]  # 论文:训练时[256, 1024]

    # ========== 模型配置 ==========
    backbone = 'dinov2'
    pine_num_classes = 4
    rubber_num_classes = 6
    num_classes = rubber_num_classes

    # ========== LAM模块配置(严格按论文) ==========
    num_tokens = 100       # 论文: m=100
    token_rank = 16        # 论文: r=16
    feature_dim = 768  # Auto-fixed to match DINOv2 output
    num_groups = 16        # 论文: G=16
    lambda_cov = 1.0       # 论文: λ=0.5
    tau = 0.5              # 论文: τ=0.5

    # DINOv2-base共12层(0-11),适配后4层
    adapt_layers = [8, 9, 10, 11]

    # ========== 输出目录配置 ==========
    output_root = os.path.join(PROJECT_ROOT, 'wood-defect-output')
    checkpoint_dir = os.path.join(output_root, 'checkpoints')
    log_dir = os.path.join(output_root, 'logs')
    result_dir = os.path.join(output_root, 'result')

    # ========== 其他配置 ==========
    # checkpoint_dir = '/home/user4/桌面/wood-defect/wood-defect-output/checkpoints'
    # log_dir = '/home/user4/桌面/wood-defect/wood-defect-output/logs'
    save_freq = 10
    eval_freq = 5

    device = 'cuda:1'
    multi_gpu = False
    gpu_ids = [1]

    # 数据增强
    use_augmentation = True
    ignore_index = 255

    pine_classes = ['background', 'dead_knot', 'sound_knot', 'missing_edge']
    rubber_classes = ['background', 'dead_knot', 'sound_knot', 'missing_edge', 'timber_core', 'crack']

    def update_for_dataset(self, dataset_name):
        """根据数据集名称更新配置"""
        if dataset_name == 'pine_wood' or dataset_name == 'pine':
            self.num_classes = self.pine_num_classes
            print(f"Updated configs for Pine Wood: {self.num_classes} classes")
        elif dataset_name == 'rubber_wood' or dataset_name == 'rubber':
            self.num_classes = self.rubber_num_classes
            print(f"Updated configs for Rubber Wood: {self.num_classes} classes")

    def create_output_dirs(self):
        """创建输出目录"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        print(f"✅ Output directories created:")
        print(f"   Checkpoints: {self.checkpoint_dir}")
        print(f"   Logs: {self.log_dir}")
        print(f"   Results: {self.result_dir}")

config = Config()