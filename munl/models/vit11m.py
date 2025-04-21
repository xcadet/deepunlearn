from thirdparty.tiny_vit import _create_tiny_vit


def create_tiny_vit_with_num_classes_and_size(num_classes: int, img_size: int):
    model_kwargs = dict(
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.1,
        img_size=img_size,
        num_classes=num_classes,
    )
    return _create_tiny_vit(
        f"tiny_vit_{img_size}_11m_{num_classes}", False, **model_kwargs
    )
