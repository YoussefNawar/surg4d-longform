import logging
from typing import List, Optional

import cv2
import numpy as np
import seaborn as sns
from utils.cholec_utils import seg8k_class_id_to_class_name

log = logging.getLogger(__name__)


def as_uint8(image: np.ndarray) -> np.ndarray:
    """Convert the image to uint8.

    Args:
        image (np.ndarray): the image to convert.

    Returns:
        np.ndarray: the converted image.
    """
    if image.dtype in [np.float16, np.float32, np.float64]:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    elif image.dtype == bool:
        image = image.astype(np.uint8) * 255
    elif image.dtype != np.uint8:
        logging.warning(f"Unknown image type {image.dtype}. Converting to uint8.")
        image = image.astype(np.uint8)
    return image


def as_float32(image: np.ndarray) -> np.ndarray:
    """Convert the image to float32.

    Args:
        image (np.ndarray): the image to convert.

    Returns:
        np.ndarray: the converted image.
    """
    if image.dtype in [np.float16, np.float32, np.float64]:
        image = image.astype(np.float32)
    elif image.dtype == bool:
        image = image.astype(np.float32)
    elif image.dtype != np.uint8:
        logging.warning(f"Unknown image type {image.dtype}. Converting to float32.")
        image = image.astype(np.float32)
    else:
        image = image.astype(np.float32) / 255
    return image


def ensure_cdim(x: np.ndarray, c: int = 3) -> np.ndarray:
    if x.ndim == 2:
        x = x[:, :, np.newaxis]
    elif x.ndim == 3:
        pass
    else:
        raise ValueError(f"bad input ndim: {x.shape}")

    if x.shape[2] < c:
        return np.concatenate([x] * c, axis=2)
    elif x.shape[2] == c:
        return x
    else:
        raise ValueError(f"bad input shape: {x.shape}")


def maybe_unsqueeze(x: np.ndarray, dim: int = 2, return_dims: bool = False):
    """Unsqueeze the input if it is less than the target dim.

    Args:
        x (np.ndarray): the input array, e.g. points.
        dim (int, optional): the target dimension. Defaults to 2.
        return_dims (bool, optional): whether to return the list of added dimensions. Defaults to False.

    Returns:
        tuple[np.ndarray, list[int]]: the unsqueezed array and the list of added dimensions.
    """
    added_dims = []
    while x.ndim < dim:
        x = np.expand_dims(x, axis=0)
        added_dims.append(0)
    if return_dims:
        return x, added_dims
    return x


def maybe_squeeze_back(x: np.ndarray, added_dims: list[int]) -> np.ndarray:
    for dim in reversed(added_dims):
        x = np.squeeze(x, axis=dim)
    return x


# TODO: redo each of these to allow for passing in a color palette and labels, as well as a scale
# factor.


def get_colors(
    num_colors: int,
    colors: Optional[np.ndarray] = None,
    palette: str = "tab20",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Get a list of colors from a seaborn palette.

    Args:
        num_colors (int): The number of colors to get.
        colors (np.ndarray, optional): An optional array of colors to use instead. Defaults to None.
        palette (str, optional): The name of the palette. Defaults to "hls".
        seed (Optional[int], optional): The seed for the random number generator. Defaults to None.

    Returns:
        np.ndarray: An array of colors, in [num_colors, 3] format, in range [0,1].
    """
    if seed is not None:
        np.random.seed(seed)

    colors_provided = False
    if colors is None:
        colors = np.array(sns.color_palette(palette, num_colors))
        colors = colors[np.random.permutation(colors.shape[0])]
    else:
        colors_provided = True
        colors = np.array(colors)

    if colors.ndim == 1:
        colors = np.tile(colors[np.newaxis, :], (num_colors, 1))

    if np.any(colors > 1):
        colors = colors.astype(float) / 255

    if not colors_provided:
        colors = colors[np.random.permutation(colors.shape[0])]
    return colors


def stack(images: list[np.ndarray], axis: int = 0):
    """Stack a list of images into a single image."""
    pad_axis = 1 - axis

    # Get the biggest image along the pad_axis
    max_size = np.max([image.shape[pad_axis] for image in images])

    # Pad on the other axis to make all images the same size
    images = [
        np.pad(
            image,
            [(0, 0)] * pad_axis
            + [(0, max_size - image.shape[pad_axis])]
            + [(0, 0)] * (2 - pad_axis),
        )
        for image in images
    ]
    log.debug(
        f"Stacking images with shapes {[image.shape for image in images]} along {axis}"
    )

    # Stack the images
    return np.concatenate(images, axis=axis)


def combine_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    channel=0,
    normalize=True,
) -> np.ndarray:
    """Visualize a heatmap on an image.

    Args:
        image (Union[np.ndarray, torch.Tensor]): 2D float image, [H, W]
        heatmap (Union[np.ndarray, torch.Tensor]): 2D float heatmap, [H, W], or [C, H, W] array of heatmaps.
        channel (int, optional): Which channel to use for the heatmap. For an RGB image, channel 0 would render the heatmap in red.. Defaults to 0.
        normalize (bool, optional): Whether to normalize the heatmap. This can lead to all-red images if no landmark was detected. Defaults to True.

    Returns:
        np.ndarray: A [H,W,3] numpy image.
    """
    image_arr = ensure_cdim(as_float32(image), c=3)
    heatmap_arr = ensure_cdim(heatmap, c=1)

    heatmap_arr = heatmap_arr.transpose(2, 0, 1)
    image_arr = image_arr.transpose(2, 0, 1)

    seg = False
    if heatmap_arr.dtype == bool:
        heatmap_arr = heatmap_arr.astype(np.float32)
        seg = True

    _, h, w = heatmap_arr.shape
    heat_sum = np.zeros((h, w), dtype=np.float32)
    for heat in heatmap_arr:
        heat_min = heat.min()
        heat_max = 4 if seg else heat.max()
        heat_min_minus_max = heat_max - heat_min
        heat = heat - heat_min
        if heat_min_minus_max > 1.0e-3:
            heat /= heat_min_minus_max

        heat_sum += heat

    for c in range(3):
        image_arr[c] = ((1 - heat_sum) * image_arr[c]) + (
            heat_sum if c == channel else 0
        )

    # log.debug(f"Combined heatmap with shape {image_arr.shape} and dtype {image_arr.dtype}")

    return as_uint8(image_arr.transpose(1, 2, 0))


def draw_heatmap(
    heatmap: np.ndarray,
    min_value: float | None = None,
    max_value: float | None = None,
    color_map: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """Draw a heatmap as an image."""
    min_val = min_value if min_value is not None else heatmap.min()
    max_val = max_value if max_value is not None else heatmap.max()
    heatmap_vis = (heatmap - min_val) / (max_val - min_val + 1e-6)
    heatmap_vis = ensure_cdim(as_uint8(heatmap_vis))
    heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_VIRIDIS)
    return heatmap_vis


def draw_corridor(
    image: np.ndarray,
    corridor: np.ndarray,
    color: np.ndarray | None = None,
    thickness: int = 1,
    palette: str = "hls",
    name: str | None = None,
) -> np.ndarray:
    """Draw a corridor on an image (copy).

    Args:
        image (np.ndarray): the image to draw on.
        corridor (np.ndarray): the corridor to draw. [2, 2] array of [x, y] coordinates.

    """

    image = draw_keypoints(image, corridor, names=["0", "1"], palette=palette)

    if color is None:
        color = np.array(sns.color_palette(palette, 1))[0]
    elif np.any(color < 1):
        color = (color * 255).astype(int)

    color_arg = np.array(color).tolist()
    log.debug(f"Drawing corridor with color: {color}")

    # Draw a line between the two points
    image = cv2.line(
        image,
        (int(corridor[0, 0]), int(corridor[0, 1])),
        (int(corridor[1, 0]), int(corridor[1, 1])),
        color_arg,
        thickness,
    )

    # Draw the name of the corridor
    if name is not None:
        fontscale = 0.75 / 512 * image.shape[0]
        thickness = max(int(1 / 256 * image.shape[0]), 1)
        offset = max(5, int(5 / 512 * image.shape[0]))
        x, y = np.mean(corridor, axis=0)
        image = cv2.putText(
            image,
            name,
            (int(x) + offset, int(y) - offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontscale,
            color_arg,
            thickness,
            cv2.LINE_AA,
        )

    return image


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    names: Optional[List[str]] = None,
    colors: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    label_order: Optional[np.ndarray] = None,
    palette: str = "hls",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw keypoints on an image (copy).

    Args:
        image (np.ndarray): the image to draw on.
        keypoints (np.ndarray): the keypoints to draw. [N, 2] array of [x, y] coordinates.
            -1 indicates no keypoint present.
        names (List[str], optional): the names of the keypoints. Defaults to None.
        colors (np.ndarray, optional): the colors to use for each keypoint. Defaults to None. If labels is provided this is overwritten.
        labels (np.ndarray, optional): the labels for each keypoint, if they should be assigned to classes. Defaults to None.
        label_order (np.ndarray, optional): the order of the labels, corresponding to colors. Defaults to None.

    Returns:
        np.ndarray: the image with the keypoints drawn.

    """

    if len(keypoints) == 0:
        return image

    if seed is not None:
        np.random.seed(seed)
    image = ensure_cdim(as_uint8(image)).copy()
    keypoints = np.array(keypoints)
    keypoints = maybe_unsqueeze(keypoints, dim=2)

    if labels is not None:
        if label_order is None:
            unique_labels = sorted(list(np.unique(labels)))
            num_colors = num_classes if num_classes is not None else len(unique_labels)
            label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        else:
            num_colors = len(label_order)
            label_to_idx = {label: i for i, label in enumerate(label_order)}
        if colors is None:
            colors = np.array(sns.color_palette(palette, num_colors))
            colors = colors[np.random.permutation(colors.shape[0])]
        colors = np.array(colors)

        colors = np.array([colors[label_to_idx[l]] for l in labels])

    if colors is None:
        colors = np.array(sns.color_palette(palette, keypoints.shape[0]))
        colors = colors[np.random.permutation(colors.shape[0])]
    else:
        colors = np.array(colors)

    if np.any(colors < 1):
        colors = (colors * 255).astype(int)

    fontscale = 0.75 / 512 * image.shape[0]
    thickness = max(int(1 / 256 * image.shape[0]), 1)
    offset = max(5, int(5 / 512 * image.shape[0]))
    radius = max(1, int(15 / 512 * image.shape[0]))

    for i, keypoint in enumerate(keypoints):
        if np.any(keypoint < 0):
            continue
        color = colors[i].tolist()
        x, y = keypoint

        # Draw a circle with a white outline.
        image = cv2.circle(image, (int(x), int(y)), radius + 1, (255, 255, 255), -1)
        image = cv2.circle(image, (int(x), int(y)), radius, color, -1)

        if names is not None:
            image = cv2.putText(
                image,
                names[i],
                (int(x) + offset, int(y) - offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness,
                cv2.LINE_AA,
            )
    return image


def outline_masks(
    image: np.ndarray,
    masks: np.ndarray,
    colors: np.ndarray,
    thickness: int = 1,
    palette: str = "hls",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Outline the masks with the given colors and alpha.

    Args:
        image (np.ndarray): the image to draw on.
        masks (np.ndarray): the masks to draw. [num_masks, H, W] array of masks.
        colors (np.ndarray): the colors to use for each mask, as an [num_masks, 3] array of RGB values in [0,255].
        thickness   (int, optional): the thickness of the outline. Defaults to 1.

    Returns:
        np.ndarray: the image with the mask outlines drawn.
    """

    image = as_float32(image).copy()
    masks = np.array(masks).astype(np.float32)
    masks = maybe_unsqueeze(masks, dim=3)

    colors = get_colors(masks.shape[0], colors=colors, palette=palette, seed=seed)

    for i, mask in enumerate(masks):
        bool_mask = mask > 0.5
        if not bool_mask.any():
            continue

        contours_, _ = cv2.findContours(
            bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        c = tuple((255 * colors[i]).astype(int).tolist())
        image = as_uint8(image)
        cv2.drawContours(image, contours_, -1, c, thickness)
        image = as_float32(image)

    return as_uint8(image)


def draw_masks(
    image: np.ndarray,
    inst_masks: np.ndarray,
    sem_masks: Optional[np.ndarray] = None,
    alpha: float = 0.3,
    threshold: float = 0.5,
    names: Optional[List[str]] = None,
    colors: Optional[np.ndarray] = None,
    name_colors: Optional[np.ndarray] = None,
    palette: str = "hls",
    seed: Optional[int] = None,
    contours: bool = True,
    contour_thickness: int = 1,
    horizontal_alignment: str = "center",
    vertical_alignment: str = "center",
    fontsize: float = 1.0,
    newlines: bool = False,
    label_mode: str = "index_and_class",
) -> np.ndarray:
    """Draw contours of masks on an image (copy).

    TODO: add options for thresholding which masks to draw based on portion of image size?

    Args:
        image (np.ndarray): the image to draw on.
        inst_masks (np.ndarray): instance-id mask map [H, W] where each nonzero id is one instance.
        sem_masks (np.ndarray, optional): semantic-id mask map [H, W].
            If provided and colors is None, instances are colored by dominant semantic overlap.
        alpha (float, optional): the alpha value for the mask. Defaults to 0.3.
        threshold (float, optional): the threshold for the mask. Defaults to 0.5.
        names (List[str], optional): the names of the masks. Defaults to None.
        colors (np.ndarray, optional): the colors to use for each mask,
            as an [num_masks, 3] array of RGB values in [0,255]. Defaults to None.
        name_colors (np.ndarray, optional): the colors to use for each mask name,
            as a [num_masks, 3] array of RGB values in [0,255]. Defaults to None.
        palette (str, optional): the name of the color palette to use. Defaults to "hls".
        seed (Optional[int], optional): the seed for the random number generator. Defaults to None.
        contours (bool, optional): whether to draw contours around the masks. Defaults to True.
        contour_thickness (int, optional): the thickness of the contours. Defaults to 1.
        horizontal_alignment (str, optional): the horizontal alignment of the names. One of "left", "center", "right". Defaults to "center".
        vertical_alignment (str, optional): the vertical alignment of the names. One of "top", "center", "bottom". Defaults to "center".
        rel_fontsize (float, optional): the relative font size of the names as a fraction of image height. Defaults to 0.05.
        label_mode (str, optional): Label text mode if names is None and sem_masks is provided.
            One of "index_and_class" or "class". Defaults to "index_and_class".

    Returns:
        np.ndarray: the image with the masks drawn.
    """

    instance_map = np.array(inst_masks)
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids != 0]
    inst_masks = [(instance_map == iid).astype(np.float32) for iid in instance_ids]
    if inst_masks:
        inst_masks = np.stack(inst_masks, axis=0)
    else:
        inst_masks = np.zeros((0, *instance_map.shape), dtype=np.float32)
    inst_masks = maybe_unsqueeze(inst_masks, dim=3)

    image = as_float32(image).copy()
    image = ensure_cdim(image)
    dominant_semantic_idx = None
    semantic_ids = None

    if sem_masks is not None:
        semantic_map = np.array(sem_masks)
        semantic_ids = np.unique(semantic_map)
        semantic_ids = semantic_ids[semantic_ids != 0]
        sem_masks = [(semantic_map == sid).astype(np.float32) for sid in semantic_ids]
        if sem_masks:
            sem_masks = np.stack(sem_masks, axis=0)
        else:
            sem_masks = np.zeros((0, *semantic_map.shape), dtype=np.float32)
        sem_masks = maybe_unsqueeze(sem_masks, dim=3)

        inst_bool_masks = inst_masks > threshold
        sem_bool_masks = sem_masks > threshold
        overlap_scores = (
            inst_bool_masks[:, np.newaxis] & sem_bool_masks[np.newaxis]
        ).sum(axis=(2, 3))
        dominant_semantic_idx = np.argmax(overlap_scores, axis=1)

        if colors is None:
            sem_colors = get_colors(
                sem_masks.shape[0], colors=None, palette=palette, seed=seed
            )
            colors = sem_colors[dominant_semantic_idx]
        else:
            colors = get_colors(
                inst_masks.shape[0], colors=colors, palette=palette, seed=seed
            )
    else:
        colors = get_colors(
            inst_masks.shape[0], colors=colors, palette=palette, seed=seed
        )

    if names is None and dominant_semantic_idx is not None:
        names = []
        for i, sem_idx in enumerate(dominant_semantic_idx):
            class_id = int(semantic_ids[sem_idx])
            class_name = seg8k_class_id_to_class_name(class_id)
            if label_mode == "index_and_class":
                label = f"{i+1}:{class_name}"
            elif label_mode == "class":
                label = class_name
            else:
                raise ValueError(f"Invalid label mode: {label_mode}")
            names.append(label)

    if name_colors is None:
        name_colors = colors
    else:
        name_colors = np.array(name_colors)
        if np.any(name_colors > 1):
            name_colors = name_colors.astype(float) / 255

    # image *= 1 - alpha
    for i, mask in enumerate(inst_masks):
        bool_mask = mask > threshold

        # if fraction > 0.9:
        #     continue
        if not bool_mask.any():
            continue

        image[bool_mask] = colors[i] * alpha + image[bool_mask] * (1 - alpha)

        if contours:
            contours_, _ = cv2.findContours(
                bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            image = as_uint8(image)
            c = tuple((255 * colors[i]).astype(int).tolist())
            cv2.drawContours(image, contours_, -1, c, contour_thickness)
            image = as_float32(image)

    image = as_uint8(image)

    # thickness = max(int(1 / 256 * image.shape[0]), 1)
    thickness = max(contour_thickness // 2, 1)
    if names is not None:
        for i, mask in enumerate(inst_masks):
            bool_mask = mask > threshold
            ys, xs = np.argwhere(bool_mask).T
            if len(ys) == 0:
                continue

            # Align the text on the mask. TODO: Make the anchor point dynamic.
            name = names[i]
            if newlines:
                name = "\n".join(name.split())
            outline_thickness = thickness + max(1, thickness // 5)
            text_size, baseline = cv2.getTextSize(
                name,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontsize,
                outline_thickness,
            )
            if horizontal_alignment == "left":
                mx = np.min(xs)
            elif horizontal_alignment == "center":
                mx = (np.min(xs) + np.max(xs)) / 2
            elif horizontal_alignment == "right":
                mx = np.max(xs)
            else:
                raise ValueError(
                    f"Invalid horizontal alignment: {horizontal_alignment}"
                )

            if vertical_alignment == "top":
                my = np.min(ys)
            elif vertical_alignment == "center":
                my = (np.min(ys) + np.max(ys)) / 2
            elif vertical_alignment == "bottom":
                my = np.max(ys)
            else:
                raise ValueError(f"Invalid vertical alignment: {vertical_alignment}")

            x = int(mx - text_size[0] // 2)
            y = int(my - text_size[1] // 2)

            x_min = 0
            x_max = max(0, image.shape[1] - text_size[0] - 1)
            y_min = text_size[1]
            y_max = max(y_min, image.shape[0] - baseline - 1)
            x = int(np.clip(x, x_min, x_max))
            y = int(np.clip(y, y_min, y_max))

            # draw the larger outline in white
            image = cv2.putText(
                image,
                name,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontsize,
                (255, 255, 255),
                outline_thickness,
                cv2.LINE_AA,
            )

            # draw the text in the mask color
            image = cv2.putText(
                image,
                name,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontsize,
                (255 * name_colors[i]).tolist(),
                thickness,
                cv2.LINE_AA,
            )

    return image


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    names: Optional[List[str]] = None,
    colors: Optional[np.ndarray] = None,
    palette: str = "hls",
    seed: Optional[int] = None,
    thickness: int = 1,
) -> np.ndarray:
    """Draw boxes on an image (copy).

    Args:
        image (np.ndarray): the image to draw on.
        boxes (np.ndarray): the boxes to draw. [N, 4] array of [x1, y1, x2, y2] coordinates.
    """

    boxes = np.array(boxes)

    image = ensure_cdim(as_uint8(image).copy())
    if colors is None:
        colors = (np.array(sns.color_palette(palette, boxes.shape[0])) * 255).astype(
            image.dtype
        )
        if seed is not None:
            np.random.seed(seed)
        colors = colors[np.random.permutation(colors.shape[0])]
    else:
        colors = np.array(colors)

    # thickness = max(int(1 / 256 * image.shape[0]), 1)
    for i, box in enumerate(boxes):
        color = colors[i].tolist()
        x1, y1, x2, y2 = box
        image = cv2.rectangle(
            image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness
        )

    fontscale = 0.75 / 512 * image.shape[0]
    offset = max(5, int(5 / 512 * image.shape[0]))

    if names is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x = x1
            y = y1 - offset
            # x = (x1 + x2) / 2
            # y = (y1 + y2) / 2
            image = cv2.putText(
                image,
                names[i],
                (int(x) + offset, int(y) - offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                (255 * colors[i]).tolist(),
                thickness,
                cv2.LINE_AA,
            )

    return image


def maybe_draw_point(
    image: np.ndarray,
    x: float | None,
    y: float | None,
    color: tuple = (0, 0, 255),
    radius: int = 20,
    label: str | None = None,
) -> np.ndarray:
    """Draws a point on an image if coordinates are valid."""
    if x is None or y is None:
        return image

    log.debug(f"Drawing point at ({x}, {y})")

    if 0 <= int(x) < image.shape[1] and 0 <= int(y) < image.shape[0]:
        cv2.circle(image, (int(x), int(y)), radius, color, -1)

        if label is not None:
            cv2.putText(
                image,
                label,
                (int(x) + radius, int(y) - radius),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                # max(int(min(image.shape) / 100),
                color,
                2,
            )

    return image


def maybe_draw_segment(
    image: np.ndarray,
    x1: float | None,
    y1: float | None,
    x2: float | None,
    y2: float | None,
    color: tuple = (0, 255, 0),
    thickness: int = 5,
    radius: int = 10,
    label: str | None = None,
) -> np.ndarray:
    """Draws a line segment on an image if coordinates are valid."""
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return image

    log.debug(f"Drawing segment from ({x1}, {y1}) to ({x2}, {y2})")

    # Draw a circle at point a
    if 0 <= int(x1) < image.shape[1] and 0 <= int(y1) < image.shape[0]:
        cv2.circle(image, (int(x1), int(y1)), radius, color, -1)

        if label is not None:
            cv2.putText(
                image,
                label,
                (int(x1) + radius, int(y1) - radius),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                color,
                2,
            )

    # Draw a circle at point b
    if 0 <= int(x2) < image.shape[1] and 0 <= int(y2) < image.shape[0]:
        cv2.circle(image, (int(x2), int(y2)), radius, color, -1)

    if (
        0 <= int(x1) < image.shape[1]
        and 0 <= int(y1) < image.shape[0]
        and 0 <= int(x2) < image.shape[1]
        and 0 <= int(y2) < image.shape[0]
    ):
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return image


def draw_circles(
    image: np.ndarray,
    centers: tuple[float, float],
    radii: float,
    colors: Optional[np.ndarray] = None,
    palette: str = "hls",
    seed: Optional[int] = None,
    contours: bool = True,
    contour_thickness: int = 2,
    alpha: float = 0.3,
) -> np.ndarray:
    """Draw a circle on an image.

    Args:
        image (np.ndarray): The image to draw on.
        centers (tuple[float, float]): The centers of the circles (x, y).
        radii (float): The radii of the circles.
        colors (Optional[np.ndarray], optional): The colors of the circles. Defaults to None.
        palette (str, optional): The color palette to use if colors is None. Defaults to "hls".
        seed (Optional[int], optional): The random seed for color selection. Defaults to None.
        contours (bool, optional): Whether to draw contours around the circles. Defaults to True.
        contour_thickness (int, optional): The thickness of the contours. Defaults to 2.
        alpha (float, optional): The alpha value for blending the circle color with the image.
        thickness (int, optional): The thickness of the circle. Defaults to 2.

    Returns:
        np.ndarray: The image with the circle drawn.
    """

    centers = maybe_unsqueeze(np.array(centers), dim=2)
    radii = maybe_unsqueeze(np.array(radii), dim=1)
    image = ensure_cdim(as_float32(image))
    colors = get_colors(
        len(centers),
        colors=colors,
        palette=palette,
        seed=seed,
    )

    for i, (center, radius) in enumerate(zip(centers, radii)):
        x, y = center
        if np.isnan(x) or np.isnan(y) or np.isnan(radius):
            continue
        if not (0 <= int(x) < image.shape[1] and 0 <= int(y) < image.shape[0]):
            continue
        if radius <= 0:
            continue
        color = colors[i]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(radius), 1, -1)
        bool_mask = mask.astype(bool)

        image[bool_mask] = color * alpha + image[bool_mask] * (1 - alpha)

    image = as_uint8(image)
    if not contours:
        return image

    for i, (center, radius) in enumerate(zip(centers, radii)):
        color = (255 * colors[i]).astype(int).tolist()
        if contours:
            image = cv2.circle(
                image,
                (int(center[0]), int(center[1])),
                int(radius),
                color,
                contour_thickness,
            )

    return image


def draw_outlined_text(
    image: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.6,
    thickness: int = 1,
    color: tuple[int, int, int] = (255, 255, 255),
    outline_color: tuple[int, int, int] = (0, 0, 0),
    outline_thickness: int = 3,
) -> np.ndarray:
    """Draw text with an outline by drawing it twice."""
    # Draw the outline first
    cv2.putText(
        image,
        text,
        pos,
        font,
        font_scale,
        outline_color,
        outline_thickness,
        cv2.LINE_AA,
    )
    # Draw the main text on top
    cv2.putText(image, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)
    return image
