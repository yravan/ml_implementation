"""
Composition Transforms
======================

Transforms for composing multiple transforms together.
"""

import numpy as np
from typing import List, Callable, Optional
import random


class Compose:
    """
    Compose multiple transforms together.

    Args:
        transforms: List of transforms to apply in order

    Example:
        >>> transform = Compose([
        ...     Resize(256),
        ...     CenterCrop(224),
        ...     ToTensor(),
        ...     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ... ])
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


class RandomApply:
    """
    Apply transforms with a probability.

    Args:
        transforms: List of transforms
        p: Probability of applying transforms
    """

    def __init__(self, transforms: List[Callable], p: float = 0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            for t in self.transforms:
                img = t(img)
        return img


class RandomChoice:
    """
    Randomly choose one transform from a list.

    Args:
        transforms: List of transforms to choose from
        p: Optional probabilities for each transform
    """

    def __init__(
        self,
        transforms: List[Callable],
        p: Optional[List[float]] = None,
    ):
        self.transforms = transforms
        self.p = p

    def __call__(self, img):
        if self.p is None:
            t = random.choice(self.transforms)
        else:
            t = random.choices(self.transforms, weights=self.p, k=1)[0]
        return t(img)


class RandomOrder:
    """
    Apply transforms in random order.

    Args:
        transforms: List of transforms
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        return img
