from pathlib import Path
from omegaconf import DictConfig
import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from typing import List, Dict, Any, Optional
import cv2
import re

import qwen_vl
from scene.dataset_readers import CameraInfo, readColmapSceneInfo
from scene.cameras import Camera


def get_patched_qwen_for_spatial_grounding(
    use_bnb_4bit: bool = False, use_bnb_8bit: bool = False
):
    model, processor = qwen_vl.get_patched_qwen(
        use_bnb_4bit=use_bnb_4bit,
        use_bnb_8bit=use_bnb_8bit,
        attn_implementation="eager",
    )

    # Enable attention output in model config
    model.config.output_attentions = True
    model.model.config.output_attentions = True
    model.model.language_model.config.output_attentions = True

    return model, processor


def extract_text_to_vision_attention(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    vision_features,
    layers: List[int],
    prompt: str,
    substring: str,
):
    """Extract attention scores from substring query tokens to vision tokens across layers.

    Args:
        model (Qwen2_5_VLForConditionalGeneration): Qwen2.5-VL model
        processor (Qwen2_5_VLProcessor): Qwen2.5-VL processor
        vision_features (_type_): _description_
        layers (List[int]): List of layers to extract attention scores from
        prompt (str): Prompt to use for the query
        substring (str): Substring to extract attention scores from

    Returns:
        Dict[str, Any]:
            - scores: torch.Tensor of shape (num_layers, num_query_tokens, num_vision_tokens)
            - tokens: List[str] of all decoded tokens for input sequence
            - query_token_indices: List[int] indices into tokens corresponding to substring span
            - vision_token_indices: List[int] indices into tokens corresponding to vision placeholders
    """
    assert substring in prompt

    # Build a message with an image placeholder and the full prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs using a mock image sized to match the number of patch features
    inputs = qwen_vl.model_inputs(messages, [vision_features], processor).to(
        model.device
    )

    with torch.no_grad():
        # Not generating any output tokens here
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
            custom_patch_features=[vision_features],
        )

    all_attn = list(outputs.attentions)

    # Identify vision token positions via <|image_pad|> placeholders
    input_ids = inputs.input_ids[0]
    image_pad_token = processor.tokenizer.encode(
        "<|image_pad|>", add_special_tokens=False
    )[0]
    image_pad_positions = (input_ids == image_pad_token).nonzero(as_tuple=True)[0]
    if image_pad_positions.numel() == 0:
        raise ValueError("No <|image_pad|> tokens found in input sequence.")
    vision_token_indices = image_pad_positions.tolist()

    # Map substring to token indices by scanning decoded tokens and locating overlap
    tokens = [processor.tokenizer.decode([tid]) for tid in input_ids.tolist()]
    concatenated = "".join(tokens)
    start_char = concatenated.lower().find(substring.lower())
    end_char = start_char + len(substring)

    query_token_indices = []
    cursor = 0
    for idx, tok in enumerate(tokens):
        token_start = cursor
        token_end = cursor + len(tok)
        if token_start < end_char and token_end > start_char:
            query_token_indices.append(idx)
        cursor = token_end
        if cursor > end_char and len(query_token_indices) > 0:
            break

    # Collect attention: [L, Q, V], averaged across heads
    num_layers = len(layers)
    num_queries = len(query_token_indices)
    num_vision = len(vision_token_indices)
    scores = torch.empty((num_layers, num_queries, num_vision), dtype=torch.float32)

    q_idx = torch.tensor(query_token_indices, device=all_attn[0].device)
    v_idx = torch.tensor(vision_token_indices, device=all_attn[0].device)

    for out_pos, layer_idx in enumerate(layers):
        layer_attn = all_attn[layer_idx]  # [1, heads, seq, seq]
        # select queries and vision columns, average heads -> [Q, V]
        attn_q_to_v = layer_attn[0, :, q_idx, :][:, :, v_idx].mean(
            dim=0
        )  # mean over heads
        scores[out_pos] = attn_q_to_v.detach().to(torch.float32).cpu()

    return {
        "scores": scores,
        "tokens": tokens,
        "query_token_indices": query_token_indices,
        "vision_token_indices": vision_token_indices,
    }


def project_3d_to_2d(
    positions: np.ndarray, proj_matrix: torch.Tensor, img_width: int, img_height: int
) -> np.ndarray:
    # Expecting positions to be (N, 3)
    assert positions.shape[1] == 3, "Positions must be (N, 3)"

    # Conver to homogeneous coordinates
    positions = torch.tensor(positions, device=proj_matrix.device, dtype=proj_matrix.dtype)
    ones = torch.ones(
        (positions.shape[0], 1), dtype=positions.dtype, device=positions.device
    )
    positions = torch.cat([positions, ones], dim=1)  # (N, 4)

    # Apply full projection transform: world to image space
    # Apparently full_proj_transform is transposed, seems to be correct
    coords = (proj_matrix.T @ positions.T).T  # (N, 4)

    # Perspective division to get NDC (Normalized Device Coordinates)
    w = coords[:, 3]
    ndc = coords[:, :3] / (w.unsqueeze(1) + 1e-7)  # (N, 3) [x, y, z] in [-1, 1]

    # Convert NDC to pixel coordinates
    # NDC: x, y in [-1, 1] → Pixel: u in [0, width], v in [0, height]
    pixels_x = (ndc[:, 0] + 1.0) * 0.5 * img_width
    pixels_y = (ndc[:, 1] + 1.0) * 0.5 * img_height

    pixels = np.stack([pixels_x, pixels_y], axis=-1)  # (N, 2)
    return pixels


def get_proj_matrix_from_timestep(
    timestep: int, train_cameras: list, frame: str
) -> torch.Tensor:
    # Get the camera parameters for the timestep
    camera_info = train_cameras[timestep]
    assert isinstance(camera_info, CameraInfo), (
        "camera_info must be a CameraInfo object"
    )

    # Instantiate a Camera object from the camera info
    image = camera_info.image
    R = camera_info.R
    T = camera_info.T
    FovX = camera_info.FovX
    FovY = camera_info.FovY
    time = camera_info.time
    mask = camera_info.mask
    camera = Camera(
        colmap_id=timestep,
        R=R,
        T=T,
        FoVx=FovX,
        FoVy=FovY,
        image=image,
        gt_alpha_mask=None,
        image_name=f"{frame}",
        uid=timestep,
        data_device=torch.device("cuda"),
        time=time,
        mask=mask,
    )

    # Get projection matrix from camera object
    # full_proj_transform includes the world to cam as well, seems to be correct
    # projection_matrix = camera.projection_matrix
    full_proj_matrix = camera.full_proj_transform
    return full_proj_matrix, camera.image_width, camera.image_height


def splat_predict_query_list(
    queries_list,
    *,
    model,
    processor,
    ts_feats,
    pos_t,
    layers,
    top_k,
    frame_number,
    clip_name: str,
    train_cameras,
    prompt_template: str,
):
    outputs = []
    for query in queries_list:
        substring = query["query"]
        prompt = prompt_template.format(substring=substring)
        attn_out = extract_text_to_vision_attention(
            model=model,
            processor=processor,
            vision_features=torch.tensor(ts_feats, device=model.device),
            layers=layers,
            prompt=prompt,
            substring=substring,
        )
        attn_scores = attn_out["scores"]

        out_item = {"query": substring, "predictions": {}}
        for layer_idx, layer in enumerate(layers):
            layer_scores = attn_scores[layer_idx]
            layer_scores = layer_scores.mean(dim=0)
            top_scores, top_indices = layer_scores.topk(k=top_k, sorted=True)
            top_scores = top_scores.detach().cpu().numpy()
            top_indices = top_indices.detach().cpu().numpy()

            top_positions = pos_t[top_indices]

            # Use frame_number directly from GT (assumed local zero-based index)
            local_idx = int(frame_number)
            frame_name = f"frame_{local_idx:06d}.jpg"
            proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
                local_idx, train_cameras, frame_name
            )
            top_pixels = project_3d_to_2d(
                top_positions, proj_matrix, img_width, img_height
            )

            out_item["predictions"][layer] = {
                "scores": top_scores.tolist(),
                "pixel_coords": top_pixels.tolist(),
                "positions": top_positions.tolist(),
            }
        outputs.append(out_item)
    return outputs


def splat_feat_queries(
    model,
    processor,
    splat_feats,
    splat_indices,
    positions,
    clip_gt,
    clip: DictConfig,
    cfg: DictConfig,
):
    # load cameras
    scene_info = readColmapSceneInfo(
        Path(cfg.preprocessed_root) / clip.name, images=None, eval=False
    )
    train_cameras = scene_info.train_cameras

    results = {}
    for timestep, timestep_queries in clip_gt.items():
        t = int(timestep)
        ts_feats = splat_feats[t]
        pos_t = positions[t][splat_indices]

        results[timestep] = {"objects": [], "actions": []}

        layers = cfg.eval.spatial.layers
        top_k = cfg.eval.spatial.top_k_scores
        frame_number = timestep_queries["frame_number"]
        prompt_template = cfg.eval.spatial.splat_prompt_template

        results[timestep]["objects"] = splat_predict_query_list(
            timestep_queries.get("objects", []),
            model=model,
            processor=processor,
            ts_feats=ts_feats,
            pos_t=pos_t,
            layers=layers,
            top_k=top_k,
            frame_number=frame_number,
            clip_name=clip.name,
            train_cameras=train_cameras,
            prompt_template=prompt_template,
        )

        results[timestep]["actions"] = splat_predict_query_list(
            timestep_queries.get("actions", []),
            model=model,
            processor=processor,
            ts_feats=ts_feats,
            pos_t=pos_t,
            layers=layers,
            top_k=top_k,
            frame_number=frame_number,
            clip_name=clip.name,
            train_cameras=train_cameras,
            prompt_template=prompt_template,
        )

    return results


def _parse_point_from_json(text: str) -> Optional[List[float]]:
    """Extract a single 3D point [x, y, z] from a JSON object in the given text.

    The model is instructed to return pure JSON, but we make this robust by:
      1) locating the first JSON object in the text, attempting json.loads
      2) falling back to regex-based triple float extraction if needed
    """
    import json as _json
    import re as _re

    # Try to find a JSON object in the text
    try:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = text[first_brace : last_brace + 1]
            obj = _json.loads(candidate)
            # Accept either {"x":..,"y":..,"z":..} or {"point":{"x":..,"y":..,"z":..}}
            if isinstance(obj, dict):
                if all(k in obj for k in ("x", "y", "z")):
                    return [float(obj["x"]), float(obj["y"]), float(obj["z"])]
                if "point" in obj and isinstance(obj["point"], dict):
                    point = obj["point"]
                    if all(k in point for k in ("x", "y", "z")):
                        return [float(point["x"]), float(point["y"]), float(point["z"])]
    except Exception:
        pass

    # Fallback: extract first three floats
    nums = _re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(nums) >= 3:
        try:
            return [float(nums[0]), float(nums[1]), float(nums[2])]
        except Exception:
            return None
    return None


def _format_only_substring(template: str, substring: str) -> str:
    """Safely format only the {substring} placeholder, escaping all other braces.

    This allows examples like {"x": 0.1} in the prompt without triggering str.format
    KeyErrors. Usage: question = _format_only_substring(tmpl, substring)
    """
    # First escape all braces
    safe = template.replace("{", "{{").replace("}", "}}")
    # Then unescape the placeholder we want to actually format
    safe = safe.replace("{{substring}}", "{substring}")
    return safe.format(substring=substring)


def static_graph_predict_query_list(
    queries_list,
    *,
    model,
    processor,
    node_feats_npz,
    adjacency_matrices: np.ndarray,
    node_centers: np.ndarray,
    node_centroids: np.ndarray,
    node_extents: np.ndarray,
    timestep_idx: int,
    frame_number: int,
    train_cameras,
    system_prompt: str,
    prompt_template: str,
):
    """Predict a single 3D point per query via Qwen prompted with a static graph.

    Returns same structure as splat_predict_query_list but without scores and with
    exactly one point per mock layer.
    """
    outputs = []

    # Precompute projection for this frame
    frame_name = f"frame_{int(frame_number):06d}.jpg"
    proj_matrix, img_width, img_height = get_proj_matrix_from_timestep(
        int(frame_number), train_cameras, frame_name
    )

    for query in queries_list:
        substring = query["query"]
        question = _format_only_substring(prompt_template, substring)

        # Call Qwen with the static graph at this timestep
        response = qwen_vl.prompt_with_static_graph(
            question=question,
            node_feats=node_feats_npz,
            node_feats_timestep_idx=int(timestep_idx),
            adjacency_matrices=adjacency_matrices,
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            model=model,
            processor=processor,
            system_prompt=system_prompt,
        )

        point3d = _parse_point_from_json(response)
        if point3d is None:
            # Fallback to origin if parsing fails
            point3d = [0.0, 0.0, 0.0]

        # Project to pixel coords
        pos_arr = np.array(point3d, dtype=np.float32).reshape(1, 3)
        pixels = project_3d_to_2d(pos_arr, proj_matrix, img_width, img_height)

        out_item = {"query": substring, "predictions": {}, "raw_response": response}
        # Single mock layer key for static baseline
        out_item["predictions"]["0"] = {
            "pixel_coords": pixels.tolist(),
            "positions": pos_arr.tolist(),
        }
        outputs.append(out_item)

    return outputs


def static_graph_feat_queries(
    *,
    model,
    processor,
    graph_dir: Path | str,
    clip_gt: Dict[str, Any],
    clip: DictConfig,
    cfg: DictConfig,
):
    """Run static-graph prompting baseline across all queries for a clip.

    Loads static graph artifacts and returns grouped results per timestep
    with the same structure as splat_feat_queries (sans scores).
    """
    graph_dir = Path(graph_dir)

    # Required static graph artifacts
    node_feats_npz_path = graph_dir / "c_qwen_feats.npz"
    adjacency_path = graph_dir / "graph.npy"
    centers_path = graph_dir / "c_centers.npy"
    centroids_path = graph_dir / "c_centroids.npy"
    extents_path = graph_dir / "c_extents.npy"

    node_feats_npz = np.load(node_feats_npz_path)
    adjacency_matrices = np.load(adjacency_path)
    node_centers = np.load(centers_path)
    node_centroids = np.load(centroids_path)
    node_extents = np.load(extents_path)

    # Cameras for projection
    scene_info = readColmapSceneInfo(
        Path(cfg.preprocessed_root) / clip.name, images=None, eval=False
    )
    train_cameras = scene_info.train_cameras

    system_prompt = cfg.eval.spatial.static_graph_system_prompt
    prompt_template = cfg.eval.spatial.static_graph_prompt_template

    results: Dict[str, Any] = {}
    for timestep, timestep_queries in clip_gt.items():
        t = int(timestep)
        frame_number = int(timestep_queries["frame_number"])  # local idx

        results[timestep] = {"objects": [], "actions": []}

        results[timestep]["objects"] = static_graph_predict_query_list(
            timestep_queries.get("objects", []),
            model=model,
            processor=processor,
            node_feats_npz=node_feats_npz,
            adjacency_matrices=adjacency_matrices,
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            timestep_idx=t,
            frame_number=frame_number,
            train_cameras=train_cameras,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
        )

        results[timestep]["actions"] = static_graph_predict_query_list(
            timestep_queries.get("actions", []),
            model=model,
            processor=processor,
            node_feats_npz=node_feats_npz,
            adjacency_matrices=adjacency_matrices,
            node_centers=node_centers,
            node_centroids=node_centroids,
            node_extents=node_extents,
            timestep_idx=t,
            frame_number=frame_number,
            train_cameras=train_cameras,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
        )

    return results


def dump_spatial_prediction_visualizations(
    *,
    results_splat: Dict[str, Any],
    clip_name: str,
    preprocessed_root: Path | str,
    images_subdir: str,
    gt_data: Dict[str, Any],
    viz_dir: Path | str,
    method_name: str | None = None,
) -> None:
    """Render top-k predicted points onto the corresponding frames and save images.

    Args:
        results_splat: Predictions dictionary returned by splat_feat_queries for a clip.
        clip_name: Name of the clip.
        preprocessed_root: Root directory for preprocessed data.
        images_subdir: Subdirectory name containing images for the clip.
        gt_data: Ground-truth dict used during evaluation; must contain frame_number per timestep.
        viz_dir: Output directory root for visualizations.
    """

    def _sanitize_filename(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^a-z0-9._-]", "", text)
        return text[:120] if len(text) > 120 else text

    def _draw_points(
        img_bgr,
        coords,
        color_bgr=(255, 0, 0),
        radius: int = 5,
        draw_indices: bool = False,
    ):
        for idx, (x, y) in enumerate(coords):
            xi, yi = int(x), int(y)
            cv2.circle(img_bgr, (xi, yi), radius, color_bgr, thickness=-1)
            if draw_indices:
                # 1-based rank index
                label = str(idx + 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                # Position to the top-right of the point
                tx = xi + radius + 3
                ty = yi - radius - 3
                # Ensure within image bounds
                tx = max(0, min(tx, img_bgr.shape[1] - tw - 1))
                ty = max(th + 1, min(ty, img_bgr.shape[0] - 1))
                # Draw background rectangle for contrast
                cv2.rectangle(
                    img_bgr,
                    (tx - 2, ty - th - 2),
                    (tx + tw + 2, ty + baseline + 2),
                    (0, 0, 0),
                    thickness=-1,
                )
                # Draw text
                cv2.putText(img_bgr, label, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return img_bgr

    # Organize outputs under .../viz_dir/<method>/<clip>
    viz_root = Path(viz_dir) / method_name / clip_name
    viz_root.mkdir(parents=True, exist_ok=True)

    images_dir = Path(preprocessed_root) / clip_name / images_subdir

    for timestep, group_preds in results_splat.items():
        timestep_str = str(timestep)
        if timestep_str not in gt_data:
            continue
        frame_number = int(gt_data[timestep_str]["frame_number"])  # type: ignore[index]
        # Use frame_number directly from GT (assumed local zero-based index)
        frame_path = images_dir / f"frame_{frame_number:06d}.jpg"
        if not frame_path.exists():
            continue
        base_img = cv2.imread(str(frame_path))
        if base_img is None:
            continue

        # Draw for objects and actions separately
        for group_name, color in (("objects", (255, 0, 0)), ("actions", (0, 0, 255))):
            items = group_preds.get(group_name, [])
            for item in items:
                query = item.get("query", group_name)
                preds_by_layer = item.get("predictions", {})
                for layer_key, pred in preds_by_layer.items():
                    layer_str = str(layer_key)
                    coords = pred.get("pixel_coords", [])
                    if not coords:
                        continue
                    img = base_img.copy()
                    img = _draw_points(img, coords, color_bgr=color, radius=5, draw_indices=True)
                    out_name = f"{frame_number:06d}_L{layer_str}_{group_name}_{_sanitize_filename(query)}.jpg"
                    cv2.imwrite(str(viz_root / out_name), img)