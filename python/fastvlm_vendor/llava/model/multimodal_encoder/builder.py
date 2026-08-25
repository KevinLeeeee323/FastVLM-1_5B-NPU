from .mobileclip_encoder import MobileCLIPVisionTower


def build_vision_tower(config, **kwargs):
    name = getattr(config, "mm_vision_tower", getattr(config, "vision_tower", ""))
    if "mobileclip" not in name.lower():
        raise ValueError("This standalone project supports only FastVLM MobileCLIP: {}".format(name))
    return MobileCLIPVisionTower(name, args=config, **kwargs)
