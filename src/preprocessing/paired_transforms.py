"""Paired spatial crops and geometric transformations for video deblurring."""
from typing import List, Tuple, Optional, Union, Any
import random
import numpy as np
from PIL import Image


def paired_random_crop(
    blur_frames: List[Union[Image.Image, np.ndarray]],
    sharp_frames: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]],
    crop_size: Tuple[int, int] = (256, 256),
    seed: Optional[int] = None
) -> Tuple[List[Union[Image.Image, np.ndarray]], Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]]]:
    """
    Apply identical spatial random crop to a list of blurred frames and corresponding sharp frame(s).
    Crop coordinates (top, left) are computed once and applied to both.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    first_frame = blur_frames[0]
    if isinstance(first_frame, Image.Image):
        w, h = first_frame.size
    elif isinstance(first_frame, np.ndarray):
        h, w = first_frame.shape[:2]
    else:
        raise TypeError(f"Unsupported frame type: {type(first_frame)}")

    crop_h, crop_w = crop_size
    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop size ({crop_h}, {crop_w}) exceeds image size ({h}, {w})")

    top = rng.randint(0, h - crop_h)
    left = rng.randint(0, w - crop_w)

    def crop_item(item):
        if isinstance(item, Image.Image):
            return item.crop((left, top, left + crop_w, top + crop_h))
        elif isinstance(item, np.ndarray):
            return item[top : top + crop_h, left : left + crop_w]
        return item

    cropped_blur = [crop_item(f) for f in blur_frames]

    if isinstance(sharp_frames, list):
        cropped_sharp = [crop_item(f) for f in sharp_frames]
    else:
        cropped_sharp = crop_item(sharp_frames)

    return cropped_blur, cropped_sharp


def paired_horizontal_flip(
    blur_frames: List[Union[Image.Image, np.ndarray]],
    sharp_frames: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]]
) -> Tuple[List[Union[Image.Image, np.ndarray]], Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]]]:
    """Flip all blurred and sharp frames horizontally."""
    def flip_item(item):
        if isinstance(item, Image.Image):
            return item.transpose(Image.FLIP_LEFT_RIGHT)
        elif isinstance(item, np.ndarray):
            return np.fliplr(item).copy()
        return item

    flipped_blur = [flip_item(f) for f in blur_frames]
    if isinstance(sharp_frames, list):
        flipped_sharp = [flip_item(f) for f in sharp_frames]
    else:
        flipped_sharp = flip_item(sharp_frames)

    return flipped_blur, flipped_sharp


def paired_vertical_flip(
    blur_frames: List[Union[Image.Image, np.ndarray]],
    sharp_frames: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]]
) -> Tuple[List[Union[Image.Image, np.ndarray]], Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]]]:
    """Flip all blurred and sharp frames vertically."""
    def flip_item(item):
        if isinstance(item, Image.Image):
            return item.transpose(Image.FLIP_TOP_BOTTOM)
        elif isinstance(item, np.ndarray):
            return np.flipud(item).copy()
        return item

    flipped_blur = [flip_item(f) for f in blur_frames]
    if isinstance(sharp_frames, list):
        flipped_sharp = [flip_item(f) for f in sharp_frames]
    else:
        flipped_sharp = flip_item(sharp_frames)

    return flipped_blur, flipped_sharp


def paired_temporal_reverse(
    blur_frames: List[Any],
    sharp_frames: Union[Any, List[Any]]
) -> Tuple[List[Any], Union[Any, List[Any]]]:
    """
    Reverse the temporal order of consecutive frames.
    For video deblurring, reversed motion is physically plausible and doubles effective sequence diversity.
    """
    rev_blur = list(reversed(blur_frames))
    if isinstance(sharp_frames, list):
        rev_sharp = list(reversed(sharp_frames))
    else:
        rev_sharp = sharp_frames
    return rev_blur, rev_sharp


class PairedTransformPipeline:
    """
    Configurable transformation pipeline for paired blurred video frames and sharp ground truth.
    Supports deterministic reproducibility with seed.
    """
    def __init__(
        self,
        crop_size: Optional[Tuple[int, int]] = (256, 256),
        flip_h: bool = True,
        flip_v: bool = True,
        temporal_reverse: bool = True,
        to_tensor_norm: bool = False,
        seed: Optional[int] = None
    ):
        self.crop_size = crop_size
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.temporal_reverse = temporal_reverse
        self.to_tensor_norm = to_tensor_norm
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def __call__(
        self,
        blur_frames: List[Image.Image],
        sharp_frames: Union[Image.Image, List[Image.Image]]
    ) -> Tuple[Any, Any]:
        b_frames = list(blur_frames)
        s_frames = list(sharp_frames) if isinstance(sharp_frames, list) else sharp_frames

        if self.crop_size is not None:
            sub_seed = self.rng.randint(0, 1000000)
            b_frames, s_frames = paired_random_crop(b_frames, s_frames, crop_size=self.crop_size, seed=sub_seed)

        if self.flip_h and self.rng.random() < 0.5:
            b_frames, s_frames = paired_horizontal_flip(b_frames, s_frames)

        if self.flip_v and self.rng.random() < 0.5:
            b_frames, s_frames = paired_vertical_flip(b_frames, s_frames)

        if self.temporal_reverse and len(b_frames) > 1 and self.rng.random() < 0.5:
            b_frames, s_frames = paired_temporal_reverse(b_frames, s_frames)

        if self.to_tensor_norm:
            def to_float(img):
                arr = np.array(img, dtype=np.float32) / 255.0
                return arr.transpose(2, 0, 1)

            b_frames = [to_float(f) for f in b_frames]
            if isinstance(s_frames, list):
                s_frames = [to_float(f) for f in s_frames]
            else:
                s_frames = to_float(s_frames)

        return b_frames, s_frames
