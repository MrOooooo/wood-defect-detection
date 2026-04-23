import os


class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PARENT_DIR = os.path.dirname(PROJECT_ROOT)

    dataset_root = os.path.join(PARENT_DIR, "pine and rubber dataset")
    pine_wood_path = os.path.join(dataset_root, "pine dataset")
    rubber_wood_path = os.path.join(dataset_root, "rubber dataset")
    dinov2_model_path = os.path.join(PARENT_DIR, "dinv2-base")

    batch_size = 4
    num_workers = 4
    num_epochs_pretrain = 10
    num_epochs_full = 20

    learning_rate = 1e-4
    # MODIFIED: keep the same learning rate in stage 2 to match the paper text.
    learning_rate_stage2 = 1e-4
    weight_decay = 0.05
    eps = 1e-8

    lr_scheduler = "poly"
    poly_power = 0.9
    min_lr = 0.0

    image_size = 512
    crop_range = [256, 1024]

    backbone = "dinov2"
    pine_num_classes = 4
    rubber_num_classes = 6
    num_classes = rubber_num_classes

    num_tokens = 100
    token_rank = 16
    # MODIFIED: this is informational only; the runtime value is inferred from the loaded backbone.
    feature_dim = 768
    num_groups = 16
    # MODIFIED: align covariance regularization weight with the paper.
    lambda_cov = 0.5
    tau = 0.5
    adapt_layers = [8, 9, 10, 11]

    output_root = os.path.join(PROJECT_ROOT, "wood-defect-output")
    checkpoint_dir = os.path.join(output_root, "checkpoints")
    log_dir = os.path.join(output_root, "logs")
    result_dir = os.path.join(output_root, "result")

    save_freq = 10
    eval_freq = 5

    device = "cuda:1"
    multi_gpu = False
    gpu_ids = [1]

    use_augmentation = True
    ignore_index = 255

    pine_classes = ["background", "dead_knot", "sound_knot", "missing_edge"]
    rubber_classes = ["background", "dead_knot", "sound_knot", "missing_edge", "timber_core", "crack"]

    def update_for_dataset(self, dataset_name):
        if dataset_name in {"pine_wood", "pine"}:
            self.num_classes = self.pine_num_classes
            print(f"Updated configs for Pine Wood: {self.num_classes} classes")
        elif dataset_name in {"rubber_wood", "rubber"}:
            self.num_classes = self.rubber_num_classes
            print(f"Updated configs for Rubber Wood: {self.num_classes} classes")

    def create_output_dirs(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        print("Output directories created:")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Logs: {self.log_dir}")
        print(f"  Results: {self.result_dir}")


config = Config()
