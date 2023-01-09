import argparse
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import seed_everything
from torchvision import transforms
from tqdm import tqdm

from src.dataloader import prepare_data
from src.stam_utils import create_pretrained_model
from src.video_model import TransformerVideoModel


def accuracy_at_k(output, target, top_k=(1, 5)):
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def parse_args():

    SUP_TRAINING = ["head", "head+partial", "head+temporal", "head+temporal-partial", "all"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--pretrained_model", type=str)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--total", type=int, default=None)

    parser.add_argument("--num_workers", type=int, default=4)

    # data stuff
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--n_frames", type=int, default=16, choices=[16, 32, 64])
    parser.add_argument("--n_clips", type=int, default=1)
    parser.add_argument("--use_queue", action="store_true")

    parser.add_argument("--corrupted", action="store_true")
    parser.add_argument("--mixed", action="store_true")

    parser.add_argument("--replace_with_mlp", action="store_true")
    parser.add_argument("--mlp_n_layers", type=int, default=1)
    parser.add_argument("--mlp_hidden_dim", type=int, default=768)
    parser.add_argument(
        "--da", type=str, default=None, choices=["cdan", "barlow", "vicreg", "simclr"]
    )
    parser.add_argument("--train", type=str, choices=SUP_TRAINING, default="head")
    parser.add_argument("--out_folder", type=str)

    args = parser.parse_args()

    # find number of classes
    args.num_classes = len(set(os.listdir(args.dataset)))
    args.classes = sorted(os.listdir(args.dataset))

    if args.corrupted:
        assert not args.mixed
    elif args.mixed:
        assert not args.corrupted

    return args


def main(args):
    device = torch.device("cuda:0")

    # load backbone and weights
    model = create_pretrained_model(args.num_classes, args.n_frames)

    model = TransformerVideoModel(model, args.num_classes, args)

    model.load_state_dict(torch.load(args.pretrained_model)["state_dict"], strict=False)

    model.to(device)
    model.eval()

    # dataloader
    dataset, _ = prepare_data(
        args.dataset,
        args.dataset,
        n_frames=args.n_frames,
        n_clips=args.n_clips,
        frame_size=args.frame_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    to_pil = transforms.ToPILImage()

    outputs = []
    targets = []
    other_images = []
    with torch.no_grad():
        for n, (x, y) in enumerate(tqdm(dataset)):
            if args.total and n > args.total:
                break

            other_images.append(x)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if args.corrupted:
                for j in range(16):
                    if random.random() < 0.1:
                        x[0, :, :, j] = -1

            elif args.mixed and n > 0:
                mixed_ids = [random.choice(range(len(other_images))) for _ in range(16)]
                for j in range(16):
                    if random.random() < 0.1:
                        mixed_x = other_images[mixed_ids[j]].to(device, non_blocking=True)
                        x[0, :, :, j] = mixed_x[0, 0, :, j]

            for clip in range(x.size(1)):
                x_clip = x[0, clip].unsqueeze(0).unsqueeze(0)

                x_clip = rearrange(x_clip, "b n_clips c f h w -> (b n_clips f) c h w")
                (feat, out), spatial_att, att = model.transformer.forward_att(x_clip)

                # spatial attention is of format frames, n_heads, tokens, tokens
                spatial_att = spatial_att[:, :, 0, 1:]
                # take mean across heads
                spatial_att = spatial_att.mean(dim=1).reshape(16, 14, 14)
                spatial_att = (
                    F.interpolate(spatial_att.unsqueeze(0), scale_factor=16, mode="nearest")[0]
                    .cpu()
                    .numpy()
                )
                att = att[0, 0, 1:]

                mean_im = torch.tensor([0.4345, 0.4051, 0.3775], device=device)
                std_im = torch.tensor([0.2768, 0.2713, 0.2737], device=device)

                fig, axs = plt.subplots(4, 4, figsize=(8, 8))

                # class_label = args.classes[int(y[0].item())]
                for i, (frame, s_a, a) in enumerate(zip(x_clip, spatial_att, att)):
                    im = to_pil(
                        frame * std_im.unsqueeze(1).unsqueeze(1) + mean_im.unsqueeze(1).unsqueeze(1)
                    )
                    # im = s_a
                    axs[i // 4, i % 4].imshow(im, aspect="auto")
                    axs[i // 4, i % 4].set_title(round(a.item(), 4), fontsize=17)
                    axs[i // 4, i % 4].set_xticks([])
                    axs[i // 4, i % 4].set_yticks([])

                plt.tick_params(
                    axis="both",
                    left="off",
                    top="off",
                    right="off",
                    bottom="off",
                    labelleft="off",
                    labeltop="off",
                    labelright="off",
                    labelbottom="off",
                )
                # plt.suptitle(f"{class_label} ({y.item()})")
                plt.tight_layout()
                pred = out.argmax(dim=1)
                fname = f"video={n}-clip={clip}-pred={pred.item()}-y={y.item()}"
                name = f"attention/{args.out_folder}/{fname}"
                if args.corrupted:
                    name += "corrupted"
                elif args.mixed:
                    name += "mixed"
                name += ".jpg"
                acc1, acc5 = accuracy_at_k(torch.cat([out]), torch.cat([y]))
                # print(
                #     clip,
                #     "yes" if acc1 > 0 else "no",
                #     "yes" if acc5 > 0 else "no",
                #     "confidence",
                #     round(out.softmax(dim=-1).max().item(), 3),
                #     "max_att",
                #     round(att.max().item(), 4),
                # )

                plt.savefig(name)
                plt.close()
                outputs.append(out)
                targets.append(y)

    outputs = torch.cat(outputs)
    targets = torch.cat(targets)

    acc1, acc5 = accuracy_at_k(outputs, targets, top_k=(1, 5))


if __name__ == "__main__":
    seed_everything(5)

    args = parse_args()

    try:
        os.makedirs(f"attention/{args.out_folder}")
    except:
        pass

    main(args)
