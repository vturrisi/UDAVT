#from ctypes import Union
from pathlib import Path
import re
from os import listdir
from os.path import join
from typing import Callable, List, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# converts string to integers when possible
def atoi(text: str):
    return int(text) if text.isdigit() else text


# applies atoi to a string
def natural_keys(text: str):
    return [atoi(c) for c in re.split(r"(\d+)", str(text))]


# returns dataset mean and std
def get_mean_std():
    mean = [0.4345, 0.4051, 0.3775]
    std = [0.2768, 0.2713, 0.2737]
    return mean, std


class VideoDataset(Dataset):
    def __init__(
        self,
        folder: str,
        n_frames: int = 16,
        n_clips: int = 1,
        frame_size: int = 224,
        reorder_shape: bool = True,
        normalize: bool = True,
    ):
        super().__init__()

        self.base_dir = folder

        self.n_frames = n_frames
        self.n_clips = n_clips
        self.reorder_shape = reorder_shape
        self.normalize = normalize
        self.mean, self.std = get_mean_std()

        if isinstance(frame_size, int):
            self.frame_size = (frame_size, frame_size)
        else:
            self.frame_size = frame_size

        self.videos_with_class = []

        self.classes = sorted(listdir(folder))

        # select all videos with enough frames
        for y, c in enumerate(self.classes):
            d = join(self.base_dir, c)
            videos = listdir(d)
            for video in videos:
                video = join(d, video)
                if len(self.find_frames(video)) >= n_frames:
                    self.videos_with_class.append((video, y))

    def __len__(self):
        return len(self.videos_with_class)

    # checks if input is image
    def is_img(self, f: Union[str, Path]):
        return str(f).lower().endswith("jpg") or str(f).lower().endswith("jpeg")

    # selects frames from input sequence
    def find_frames(self, video: Union[str, Path]):
        frames = [join(video, f) for f in listdir(video) if self.is_img(f)]
        return frames

    # handles the case where tensor was converted to gray scale
    def maybe_fix_gray(self, tensor: torch.Tensor):
        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    # generates the pipeline of transformations that we apply to an image
    def gen_transformation_pipeline(self):
        if self.frame_size[0] == 224:
            s = (256, 256)
        else:
            raise Exception("Size is not supported")

        transformations = [(TF.resize, s)]
        transformations.append((TF.center_crop, self.frame_size))

        transformations.append((TF.to_tensor,))
        transformations.append((self.maybe_fix_gray,))

        if self.normalize and self.mean is not None:
            transformations.append((TF.normalize, self.mean, self.std))
        return transformations

    # applies the generated transformations to an image
    def apply_transforms(self, frame: torch.Tensor, transformations: List[Callable]):
        for transform, *args in transformations:
            frame = transform(frame, *args)
        return frame

    # stacks the frame-level features into a video-level feature
    def convert_to_video(self, frames: torch.Tensor):
        transformations = self.gen_transformation_pipeline()

        tensors = []
        for frame in frames:
            frame = self.apply_transforms(frame, transformations)
            tensors.append(frame)

        tensors = torch.stack(tensors)
        tensors = tensors.reshape(self.n_clips, self.n_frames, *tensors.size()[1:])

        if self.reorder_shape:
            tensors = tensors.permute(0, 2, 1, 3, 4)
        return tensors

    # loads image from file
    def load_frame(self, path: Union[str, Path]):
        frame = Image.open(path)
        return frame

    # generates random indices from sequence
    def get_random_indices(self, num_frames: int):
        indexes = np.sort(np.random.choice(num_frames, self.n_frames * self.n_clips, replace=True))
        return indexes

    # generates uniformly distributed indices from sequence
    def get_indices(self, num_frames: int):
        tick = num_frames / self.n_frames
        indexes = np.array(
            [int(tick / 2.0 + tick * x) for x in range(self.n_frames)]
        )  # pick the central frame in each segment
        return indexes

    # retrieves clip indices
    def get_indices_clips(self, num_frames: int):
        num_frames_clip = num_frames // self.n_clips
        indexes = self.get_indices(num_frames_clip)
        indexes = np.tile(indexes, self.n_clips)
        for i in range(self.n_clips):
            indexes[i * self.n_frames : (i + 1) * self.n_frames] += num_frames_clip * i
        return indexes

    def __getitem__(self, index: int):
        video, y = self.videos_with_class[index]

        # find frames
        frame_paths = self.find_frames(video)
        frame_paths.sort(key=natural_keys)

        n_frames = len(frame_paths)
        indexes = self.get_indices_clips(n_frames)

        frames = []
        for i in indexes:
            frames.append(self.load_frame(frame_paths[i]))

        tensor = self.convert_to_video(frames)
        return tensor, y


class VideoDatasetSourceAndTarget:
    def __init__(self, source_dataset: Dataset, target_dataset: Dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self):
        return max([len(self.source_dataset), len(self.target_dataset)])

    def __getitem__(self, index: int):
        source_index = index % len(self.source_dataset)
        source_data = self.source_dataset[source_index]

        target_index = index % len(self.target_dataset)
        target_data = self.target_dataset[target_index]
        return (*source_data, *target_data)


# prepare datasets in the source-only case
def prepare_datasets(
    train_source_dataset: str,
    val_dataset: str,
    train_target_dataset: str = None,
    n_frames: int = 4,
    n_clips: int = 4,
    frame_size: int = 224,
    normalize: bool = True,
):
    train_dataset = VideoDataset(
        train_source_dataset,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=n_clips,
        normalize=normalize,
    )
    if train_target_dataset:
        train_target_dataset = VideoDataset(
            train_target_dataset,
            frame_size=frame_size,
            n_frames=n_frames,
            n_clips=n_clips,
            normalize=normalize,
        )
        train_dataset = VideoDatasetSourceAndTarget(train_dataset, train_target_dataset)

    val_dataset = VideoDataset(
        val_dataset,
        frame_size=frame_size,
        n_frames=n_frames,
        n_clips=1,
        normalize=normalize,
    )

    return train_dataset, val_dataset


# prepares dataloaders given input datasets
def prepare_dataloaders(
    train_dataset: str, val_dataset: str, batch_size: int = 64, num_workers: int = 4
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


# prepares datasets and dataloaders in the source-only case
def prepare_data(
    train_source_dataset: str,
    val_dataset: str,
    train_target_dataset: str = None,
    n_frames: int = 16,
    n_clips: int = 1,
    frame_size: int = 224,
    normalize: bool = True,
    batch_size: int = 64,
    num_workers: int = 4,
):
    train_dataset, val_dataset = prepare_datasets(
        train_source_dataset=train_source_dataset,
        val_dataset=val_dataset,
        train_target_dataset=train_target_dataset,
        n_frames=n_frames,
        n_clips=n_clips,
        frame_size=frame_size,
        normalize=normalize,
    )

    train_loader, val_loader = prepare_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return train_loader, val_loader
