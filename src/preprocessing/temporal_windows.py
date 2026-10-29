"""Temporal window generation utilities for video deblurring datasets."""
from typing import List, Tuple, Any, Optional
import numpy as np


def build_temporal_indices(
    sequence_length: int,
    window_size: int = 5,
    center_idx: Optional[int] = None,
    boundary_policy: str = "replicate"
) -> List[List[int]]:
    """
    Generate frame index lists of length `window_size` for every target center frame in a sequence.
    Guarantees windows never cross sequence boundaries.

    Policies for sequence boundaries:
    - 'replicate': clamp indices outside [0, sequence_length - 1] to 0 or sequence_length - 1 (e.g. [-2, -1, 0, 1, 2] -> [0, 0, 0, 1, 2]). Standard in EDVR.
    - 'reflect': reflect index across boundary (e.g. [-1, 0, 1] -> [1, 0, 1]).
    - 'truncate': only return windows where all indices strictly fall within [0, sequence_length - 1].
    - 'zero_pad': return None or -1 for out-of-bound indices so loader pads with black frame.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd for symmetric center frame target, got {window_size}")

    radius = window_size // 2
    windows = []

    # If center_idx is specified, compute only for that center frame
    targets = [center_idx] if center_idx is not None else list(range(sequence_length))

    for center in targets:
        raw_indices = [center + offset for offset in range(-radius, radius + 1)]

        if boundary_policy == "truncate":
            if any(idx < 0 or idx >= sequence_length for idx in raw_indices):
                continue
            windows.append(raw_indices)

        elif boundary_policy == "replicate":
            clamped = [max(0, min(sequence_length - 1, idx)) for idx in raw_indices]
            windows.append(clamped)

        elif boundary_policy == "reflect":
            def reflect_idx(i):
                if i < 0:
                    return -i
                if i >= sequence_length:
                    return 2 * (sequence_length - 1) - i
                return i
            reflected = [reflect_idx(idx) for idx in raw_indices]
            windows.append(reflected)

        elif boundary_policy == "zero_pad":
            padded = [idx if (0 <= idx < sequence_length) else -1 for idx in raw_indices]
            windows.append(padded)

        else:
            raise ValueError(f"Unknown boundary_policy '{boundary_policy}'. Must be one of: 'replicate', 'reflect', 'truncate', 'zero_pad'")

    return windows
