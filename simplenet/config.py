"""从 config.toml 加载参数并构建 SimpleNetConfig。"""

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .simplenet import SimpleNetConfig


def load_config(path="config.toml"):
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_simplenet_config(cfg: dict, device: str = "cuda") -> SimpleNetConfig:
    arch = cfg["architecture"]
    train = cfg["training"]
    noise = cfg["noise"]
    augment = cfg["augment"]
    misc = cfg["misc"]

    return SimpleNetConfig(
        target_size=arch["target_size"],
        layer_indices=list(arch["layer_indices"]),
        input_planes=arch["input_planes"],
        hidden_dim=arch["hidden_dim"],
        meta_epochs=train["meta_epochs"],
        gan_epochs=train["gan_epochs"],
        batch_size=train["batch_size"],
        proj_lr=train["proj_lr"],
        dsc_lr=train["dsc_lr"],
        dsc_margin=train["dsc_margin"],
        use_scheduler=train["use_scheduler"],
        noise_std=noise["noise_std"],
        mix_noise=noise["mix_noise"],
        use_augment=augment["use_augment"],
        augment_categories=list(augment["augment_categories"]),
        color_augment_categories=list(augment["color_augment_categories"]),
        patch_size=misc["patch_size"],
        resize=misc["resize"],
        isize=misc["isize"],
        device=device,
    )


def get_paths(cfg: dict) -> dict:
    return dict(cfg["paths"])
