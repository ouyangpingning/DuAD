"""从 config.toml 加载参数并构建 ModelConfig、路径、类别覆盖。"""
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from myAD import ModelConfig


def load_config(path="config.toml"):
    """加载 TOML 配置文件，返回完整 dict。"""
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_model_config(cfg: dict, device: str = "cuda") -> ModelConfig:
    """从 TOML dict 构建 ModelConfig 实例。"""
    arch = cfg["architecture"]
    train = cfg["training"]
    noise = cfg["noise"]
    pca = cfg["pca_mask"]
    perlin = cfg["perlin_mask"]
    augment = cfg["augment"]
    pca_student = cfg.get("pca_student", {})
    misc = cfg["misc"]

    return ModelConfig(
        target_size=arch["target_size"],
        layer_indices=list(arch["layer_indices"]),
        input_planes=arch["input_planes"],
        hidden_dim=arch["hidden_dim"],
        meta_epochs=train["meta_epochs"],
        gan_epochs=train["gan_epochs"],
        batch_size=train["batch_size"],
        dsc_margin=train["dsc_margin"],
        proj_lr=train["proj_lr"],
        dsc_lr=train["dsc_lr"],
        noise_std=noise["noise_std"],
        use_noise_annealing=noise["use_noise_annealing"],
        noise_std_max=noise["noise_std_max"],
        noise_std_min=noise["noise_std_min"],
        noise_anneal_type=noise["noise_anneal_type"],
        use_pca_mask=pca["use_pca_mask"],
        pca_threshold=pca["pca_threshold"],
        pca_border=pca["pca_border"],
        pca_kernel_size=pca["pca_kernel_size"],
        pca_use_gpu=pca["pca_use_gpu"],
        pca_skip_categories=list(pca["skip_categories"]),
        use_perlin_mask=perlin["use_perlin_mask"],
        perlin_min=perlin["perlin_min"],
        perlin_max=perlin["perlin_max"],
        perlin_branch_weight=perlin["perlin_branch_weight"],
        pca_branch_weight=perlin["pca_branch_weight"],
        augment_categories=list(augment["augment_categories"]),
        color_augment_categories=list(augment["color_augment_categories"]),
        patch_size=misc["patch_size"],
        use_scheduler=train["use_scheduler"],
        device=device,
        # PCA Student 配置
        use_pca_student=pca_student.get("use_pca_student", False),
        pca_student_hidden_dims=list(pca_student.get("hidden_dims", [512, 128])),
        pca_student_lr=pca_student.get("lr", 0.001),
        pca_student_epochs=pca_student.get("epochs", 50),
        pca_student_batch_size=pca_student.get("batch_size", 4096),
    )


def get_category_pca_thresholds(cfg: dict) -> dict:
    """返回类别特定的 PCA 阈值字典。"""
    return dict(cfg["category_pca"]["threshold"])


def get_category_pca_border_thresholds(cfg: dict) -> dict:
    """返回类别特定的 PCA 边界阈值字典。"""
    return dict(cfg["category_pca"]["border"])


def get_paths(cfg: dict) -> dict:
    """返回路径配置字典。"""
    return dict(cfg["paths"])
