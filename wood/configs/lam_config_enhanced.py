# configs/lam_config_enhanced.py
"""
⚠️ 增强版配置文件 - 添加跨层特征聚合参数

新增配置:
1. use_cross_layer_aggregation: 是否启用跨层聚合
2. cross_layer_output_dim: 跨层聚合输出维度
3. aggregation_method: 聚合方法
4. use_domain_alignment: 是否启用域对齐
"""

import os


class Config:
    # ========== 项目根目录 (原始,不变) ==========
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PARENT_DIR = os.path.dirname(PROJECT_ROOT)

    # ========== 数据集配置 (原始,不变) ==========
    dataset_root = os.path.join(PARENT_DIR, 'pine and rubber dataset')
    pine_wood_path = os.path.join(dataset_root, 'pine dataset')
    rubber_wood_path = os.path.join(dataset_root, 'rubber dataset')
    dinov2_model_path = os.path.join(PARENT_DIR, 'dinv2-base')

    # ========== 训练配置 (原始,不变) ==========
    batch_size = 4
    num_workers = 4
    num_epochs_pretrain = 10
    num_epochs_full = 20

    # ========== 优化器配置 (原始,不变) ==========
    learning_rate = 1e-4
    learning_rate_stage2 = 5e-5
    weight_decay = 0.05
    eps = 1e-8
    use_strong_augmentation = True

    # ========== 学习率调度器 (原始,不变) ==========
    lr_scheduler = 'poly'
    poly_power = 0.9
    min_lr = 0

    # ========== 图像配置 (原始,不变) ==========
    image_size = 512
    crop_range = [256, 1024]

    # ========== 模型配置 (原始,不变) ==========
    backbone = 'dinov2'
    pine_num_classes = 4
    rubber_num_classes = 6
    num_classes = rubber_num_classes

    # ========== LAM模块配置 (原始,不变) ==========
    num_tokens = 100
    token_rank = 16
    feature_dim = 768
    num_groups = 16
    lambda_cov = 1.0
    tau = 0.5
    adapt_layers = [8, 9, 10, 11]

    # ========== 新增: 跨层特征聚合配置 ==========
    # 是否启用跨层特征聚合
    use_cross_layer_aggregation = True  # ⭐ 新增参数

    # 跨层聚合输出维度
    cross_layer_output_dim = 512  # ⭐ 新增参数

    # 聚合方法: 'dynamic_weighted', 'attention', 'concat'
    aggregation_method = 'dynamic_weighted'  # ⭐ 新增参数

    # 是否启用域对齐(用于跨域任务)
    use_domain_alignment = False  # ⭐ 新增参数,默认关闭

    # 域对齐相关配置
    num_domain_prototypes = 10  # ⭐ 域原型数量
    domain_alignment_weight = 0.1  # ⭐ 域对齐损失权重

    # ========== 输出目录配置 (原始,不变) ==========
    output_root = os.path.join(PROJECT_ROOT, 'wood-defect-output')
    checkpoint_dir = os.path.join(output_root, 'checkpoints')
    log_dir = os.path.join(output_root, 'logs')
    result_dir = os.path.join(output_root, 'result')

    # ========== 其他配置 (原始,不变) ==========
    save_freq = 10
    eval_freq = 5
    device = 'cuda:1'
    multi_gpu = False
    gpu_ids = [1]
    use_augmentation = True
    ignore_index = 255

    pine_classes = ['background', 'dead_knot', 'sound_knot', 'missing_edge']
    rubber_classes = ['background', 'dead_knot', 'sound_knot', 'missing_edge', 'timber_core', 'crack']

    def update_for_dataset(self, dataset_name):
        """根据数据集名称更新配置 (原始,不变)"""
        if dataset_name == 'pine_wood' or dataset_name == 'pine':
            self.num_classes = self.pine_num_classes
            print(f"Updated configs for Pine Wood: {self.num_classes} classes")
        elif dataset_name == 'rubber_wood' or dataset_name == 'rubber':
            self.num_classes = self.rubber_num_classes
            print(f"Updated configs for Rubber Wood: {self.num_classes} classes")

    def create_output_dirs(self):
        """创建输出目录 (原始,不变)"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        print(f"✅ Output directories created:")
        print(f"   Checkpoints: {self.checkpoint_dir}")
        print(f"   Logs: {self.log_dir}")
        print(f"   Results: {self.result_dir}")

    # ========== 新增: 打印增强配置 ==========
    def print_enhancement_config(self):
        """打印跨层聚合配置"""
        print("\n" + "=" * 70)
        print("📊 Cross-Layer Aggregation Configuration")
        print("=" * 70)
        print(f"  Enabled: {self.use_cross_layer_aggregation}")
        if self.use_cross_layer_aggregation:
            print(f"  Output Dim: {self.cross_layer_output_dim}")
            print(f"  Aggregation Method: {self.aggregation_method}")
            print(f"  Adapted Layers: {self.adapt_layers}")
            print(f"  Number of Layers: {len(self.adapt_layers)}")
        print(f"\n  Domain Alignment: {self.use_domain_alignment}")
        if self.use_domain_alignment:
            print(f"  Num Prototypes: {self.num_domain_prototypes}")
            print(f"  Alignment Weight: {self.domain_alignment_weight}")
        print("=" * 70 + "\n")


# ========== 预定义配置模板 ==========

class BaselineConfig(Config):
    """基线配置 - 原论文设置"""
    use_cross_layer_aggregation = False
    use_domain_alignment = False


class EnhancedConfig(Config):
    """增强配置 - 启用跨层聚合"""
    use_cross_layer_aggregation = True
    cross_layer_output_dim = 512
    aggregation_method = 'dynamic_weighted'
    use_domain_alignment = False


class CrossDomainConfig(Config):
    """跨域配置 - 完整增强(CLA + 域对齐)"""
    use_cross_layer_aggregation = True
    cross_layer_output_dim = 512
    aggregation_method = 'dynamic_weighted'
    use_domain_alignment = True
    domain_alignment_weight = 0.1

    # 跨域训练特殊设置
    num_epochs_pretrain = 15  # 增加预训练轮数
    learning_rate = 5e-5  # 降低学习率以稳定训练


# ========== 默认配置实例 ==========
config = Config()  # 默认使用增强配置

# ========== 使用示例 ==========
if __name__ == "__main__":
    print("Testing Enhanced Configuration...")

    # 测试1: 基线配置
    print("\n1. Baseline Configuration (Original Paper)")
    baseline = BaselineConfig()
    baseline.print_enhancement_config()

    # 测试2: 增强配置
    print("\n2. Enhanced Configuration (Cross-Layer Only)")
    enhanced = EnhancedConfig()
    enhanced.print_enhancement_config()

    # 测试3: 跨域配置
    print("\n3. Cross-Domain Configuration (Full Enhancement)")
    cross_domain = CrossDomainConfig()
    cross_domain.print_enhancement_config()

    print("\n" + "=" * 70)
    print("✅ Configuration test passed!")
    print("=" * 70)

    print("\n📝 Usage:")
    print("from configs.lam_config_enhanced import BaselineConfig, EnhancedConfig, CrossDomainConfig")
    print("\n# 选择配置")
    print("config = EnhancedConfig()  # 或 BaselineConfig() 或 CrossDomainConfig()")