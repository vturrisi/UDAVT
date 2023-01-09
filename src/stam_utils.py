import argparse

import torch

from .stam.transformer import STAM_224


def create_pretrained_model(n_classes: int, n_frames: int, path=None):
    args_dict = {
        "model_name": "stam_{}".format(n_frames),
        "num_classes": 400,
        "input_size": 224,
        "val_zoom_factor": 0.875,
    }
    args = argparse.Namespace(**args_dict)

    args.model_name = args.model_name.lower()

    if args.model_name == "stam_16":
        args.frames_per_clip = 16
    elif args.model_name == "stam_32":
        args.frames_per_clip = 32
        args.frame_rate = 3.2
    elif args.model_name == "stam_64":
        args.frames_per_clip = 64
        args.frame_rate = 6.4
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    model_params = {"args": args, "num_classes": args.num_classes}
    model = STAM_224(model_params)
    if path:
        ckpt = torch.load(path, map_location="cpu")["model"]
        model.load_state_dict(ckpt, strict=False)

    model.reset_classifier(n_classes)
    return model
