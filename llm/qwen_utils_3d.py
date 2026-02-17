"""
Utility functions for loading and using Qwen3-VL models with 3D Gaussian-to-grid positional encoding.

Provides model loading function that returns CustomQwen3VLForConditionalGeneration3D
with support for many-to-one Gaussian-to-grid mapping.
"""

import torch
from typing import Any, Dict, Literal, Optional, Union
from transformers import Qwen3VLProcessor
from typing import List

from .patched_qwen_3d import (
    CustomQwen3VLForConditionalGeneration3D,
)
from .qwen_utils import closest_factor_pair, QWEN_CONSTANTS
from PIL import Image

import logging
logger = logging.getLogger(__name__)



def get_custom_qwen3_3d(
    size: Literal["8B", "32B"] = "8B",
    use_fp8: bool = False,
    attn_implementation: str = "sdpa",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Union[str, Dict[str, str]] = "auto",
    max_memory: Optional[Dict[str, str]] = None,
    repetition_penalty: float = None,
    compile: bool = False,
):
    """Get a custom Qwen3VL model with 3D Gaussian-to-grid positional encoding support.

    Loads the base Qwen3-VL model and swaps the class to CustomQwen3VLForConditionalGeneration3D
    which supports many-to-one Gaussian-to-grid mapping for 3D spatial queries.

    Args:
        size: Model size ("8B" or "32B")
        use_fp8: Whether to use FP8 quantized version
        attn_implementation: Attention implementation ("sdpa" or "flash_attention_2")
        torch_dtype: Data type for model weights
        device_map: Device mapping strategy
        max_memory: Maximum memory per device
        repetition_penalty: Repetition penalty for generation
        compile: Whether to compile the model with torch.compile

    Returns:
        Tuple of (model, processor) where:
            model: CustomQwen3VLForConditionalGeneration3D instance
            processor: Qwen3VLProcessor instance
    """
    model_path = f"Qwen/Qwen3-VL-{size.upper()}-Thinking"
    if use_fp8:
        model_path = model_path + "-FP8"

    fp_kwargs: Dict[str, Any] = {
        "dtype": torch_dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
    }
    if max_memory is not None:
        fp_kwargs["max_memory"] = max_memory

    # Load base model using from_pretrained
    # We load as the base class first, then swap to our custom class
    from transformers import Qwen3VLForConditionalGeneration
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        **fp_kwargs,
    )

    # Swap to our custom class that supports 3D positional encoding
    model.__class__ = CustomQwen3VLForConditionalGeneration3D
    # Re-initialize to ensure the inner model is also swapped
    # (CustomQwen3VLForConditionalGeneration3D.__init__ does this, but we need to call it)
    # Actually, since we're swapping the class, we need to ensure the inner model is also swapped
    # The __init__ method in CustomQwen3VLForConditionalGeneration3D does: self.model.__class__ = CustomQwen3VLModel3D
    # But we're swapping after __init__, so we need to do it manually
    from .patched_qwen_3d import CustomQwen3VLModel3D
    model.model.__class__ = CustomQwen3VLModel3D

    processor = Qwen3VLProcessor.from_pretrained(model_path)
    
    # Configure generation settings
    model.generation_config.return_legacy_cache = False
    if repetition_penalty is not None:
        model.generation_config.repetition_penalty = repetition_penalty
    
    model.eval()
    
    if compile:
        model = torch.compile(model, mode="reduce-overhead")
    
    return model, processor


def model_inputs(
    messages: List[Dict[str, Any]],
    vision_features: List[torch.Tensor],
    processor: Qwen3VLProcessor,
    image_token_id: Optional[int] = None,
    video_token_id: Optional[int] = None,
    image_pad_token_id: Optional[int] = None,
    vision_start_token_id: Optional[int] = None,
    vision_end_token_id: Optional[int] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    gaussians_per_image: Optional[List[int]] = None,
    tools: List[Dict[str, Any]] = [],
    **kwargs,
):
    """Prepare model inputs from messages and vision features.
    
    Args:
        messages: Chat messages with image placeholders
        vision_features: List of vision feature tensors
        processor: Qwen3VLProcessor instance
        image_token_id: Token ID for <image> token
        image_pad_token_id: Token ID for <image_pad> token
        vision_start_token_id: Token ID for <vision_start> token
        vision_end_token_id: Token ID for <vision_end> token
        image_grid_thw: Grid dimensions (BEFORE spatial merge)
        gaussians_per_image: List of number of Gaussians per image.
            If provided, adjusts input_ids to have the correct number of <image> tokens.
        tools: Optional tools for chat template
    """

    # create actual text template from messages dict
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=tools
    )
    
    logger.info("=" * 80)
    logger.info("model_inputs: CHAT TEMPLATE OUTPUT")
    logger.info("=" * 80)
    logger.info(f"Generated text length: {len(text)} chars")
    logger.info(f"Generated text (first 1000 chars): {text[:1000]}")
    logger.info(f"Generated text (last 1000 chars): {text[-1000:]}")
    if len(text) > 2000:
        logger.warning(f"⚠️ TEXT IS VERY LONG: {len(text)} chars - may be truncated!")

    # create mock images such that their size corresponds to
    # the correct number of tokens after tokenizing and spatial merging
    # (we cannot just overwrite image_grid_thw because
    # input ids already contains placeholder vision tokens)
    effective_patch_size = QWEN_CONSTANTS["qwen3"]["effective_patch_size"]
    temporal_patch_size = QWEN_CONSTANTS["qwen3"]["temporal_patch_size"]
    if image_grid_thw is not None:
        assert image_grid_thw.shape[0] == len(vision_features)
    if video_grid_thw is not None:
        assert video_grid_thw.shape[0] == len(vision_features)
    mock_images: List[Image.Image] = []
    for i, feat in enumerate(vision_features):
        if image_grid_thw is None and video_grid_thw is None:
            n = int(feat.shape[0])
            w, h = closest_factor_pair(n)
            img_w = w * effective_patch_size
            img_h = h * effective_patch_size
        elif image_grid_thw is not None:
            height = int(image_grid_thw[i, 1].item())
            width = int(image_grid_thw[i, 2].item())
            img_w = width * effective_patch_size
            img_h = height * effective_patch_size
        else:
            height = int(video_grid_thw[i, 1].item())
            width = int(video_grid_thw[i, 2].item())
            logger.info(f"Desired video grid thw spatial resolution: {height}, {width})")
            img_w = width * effective_patch_size
            img_h = height * effective_patch_size
            logger.info(f"Video grid spatial resolution after accounting for effective patch size: ({img_w}, {img_h})")
        logger.info(f"Mock image {i}: ({img_w}, {img_h}) pixels, features: {feat.shape[0]}")
        mock_img = Image.new("RGB", (img_w, img_h), color="red")
        mock_images.append(mock_img)

    has_video = any(
        part.get("type") == "video"
        for msg in messages
        if msg["role"] == "user"
        for part in msg.get("content", [])
    )
    if has_video:
        # Preserve per-frame shape information by passing a frame list for one video.
        # TODO: this is accounting for temporal patch size, either use the Qwen constant or move this to hydra
        video_input = [list(mock_images) + list(mock_images)]
        logger.info(
            f"video_input structure: {len(video_input)} video(s), "
            f"{len(video_input[0]) if video_input else 0} frame(s) in first video"
        )
        logger.info(f"number of images: {len(gaussians_per_image)}")
        inputs = processor(
            text=[text],
            videos=video_input,
            padding=False,
            return_tensors="pt",
            do_sample_frames=False,
            do_resize=False,
        )
    else:
        # This will compute the image grid thw based on (mock) image dims and account for patch size only (no spatial merge yet)
        # That means the returned image grid thw will be the desired grid size divided by patch size but before spatial merge, still "too large"
        # Later, a spatial feature merge would be performed (we do not do this on Gaussian tokens) -> grid is divided by spatial merge factor in rope index computation in model forward pass
        # -> yields the actual LLM grid size we want
        # In a sense, we want image grid thw after all this -> above we multiply by effective patch size = patch size * spatial merge factor
        # -> processor divides by patch size, and rope index computation divides by spatial merge -> reverted to desired grid size for final inference
        inputs = processor(
            text=text,
            images=mock_images,
            padding=False,
            return_tensors="pt",
            do_resize=False,
        )
    
    logger.info(f"After processor: input_ids shape: {inputs['input_ids'].shape}")
    logger.info(f"After processor: attention_mask shape: {inputs['attention_mask'].shape if 'attention_mask' in inputs else None}")

    # Decode all input ids after processor and print without skipping special tokens
    input_ids = inputs["input_ids"]
    logger.info(f"input_ids shape: {input_ids.shape}")
    torch.set_printoptions(threshold=10000)
    decoded_input = processor.tokenizer.decode(input_ids[0], skip_special_tokens=False)
    logger.info(f"Decoded input: {decoded_input}")
    
    # Adjust input_ids to allow for passing the Gaussian tokens; processor will assume a 1:1 mapping between tokens derived from llm grid and actual tokens
    # In our case however, we have a desired grid size but we might have a many : 1 mapping between actual tokens and llm grid cells
    # This is still distinct from the actual assignment of tokens to grid cells for pos enc, which is handled in the rope index computation
    # Note: this is called already with the filtered, final selection of gaussian tokens!
    if gaussians_per_image is not None:
        assert image_token_id is not None
        assert video_token_id is not None
        assert image_pad_token_id is not None
        assert vision_start_token_id is not None
        assert vision_end_token_id is not None
        assert len(gaussians_per_image) == len(vision_features), (
            f"gaussians_per_image length ({len(gaussians_per_image)}) must match "
            f"vision_features length ({len(vision_features)})"
        )
        logger.info(f"Adjusting input_ids for gaussians_per_image: {gaussians_per_image}")
        inputs = adjust_input_ids_for_gaussians(
            inputs,
            processor,
            gaussians_per_image,
            image_token_id,
            video_token_id,
            image_pad_token_id,
            vision_start_token_id,
            vision_end_token_id,
        )
        logger.info(f"After adjustment: input_ids shape: {inputs['input_ids'].shape}")

    return inputs


def adjust_input_ids_for_gaussians(
    inputs: Dict[str, torch.Tensor],
    processor: Qwen3VLProcessor,
    gaussians_per_image: List[int],
    image_token_id: int,
    video_token_id: int,
    image_pad_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Adjust input_ids to have the correct number of visual tokens per frame/image.
    
    Uses the same logic as get_placeholder_mask: finds all <image> tokens directly,
    groups consecutive ones into blocks (one per image), and replaces each block
    with the correct number of <image> tokens to match num_gaussians.
    
    Args:
        inputs: Dictionary from processor containing input_ids, attention_mask, etc.
        processor: Qwen3VLProcessor instance
        gaussians_per_image: List of number of Gaussians per image
        image_token_id: Token ID for <image> token
        video_token_id: Token ID for <video> token
    
    Returns:
        Modified inputs dict with adjusted input_ids and attention_mask
    """

    input_ids = inputs["input_ids"].clone()  # (B, L)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.clone()

    logger.info(f"input_ids of shape: {input_ids.shape}")
    logger.info(f"attention_mask of shape: {attention_mask.shape}")
    logger.info(f"image_token_id: {image_token_id}")
    logger.info(f"video_token_id: {video_token_id}")

    batch_size = input_ids.shape[0]
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert len(gaussians_per_image) > 0, "gaussians_per_image must not be empty"
    
    for b in range(batch_size):
        seq = input_ids[b]  # (L,)
        
        # Find all visual token positions (same logic as get_placeholder_mask)
        visual_mask = (seq == image_token_id) | (seq == video_token_id)
        visual_positions = visual_mask.nonzero(as_tuple=True)[0]

        # Do the same counting for the other provided token ids
        vision_start_mask = seq == vision_start_token_id
        vision_start_positions = vision_start_mask.nonzero(as_tuple=True)[0]
        vision_end_mask = seq == vision_end_token_id
        vision_end_positions = vision_end_mask.nonzero(as_tuple=True)[0]

        logger.info(f"Found total of {len(vision_start_positions)} vision start tokens")
        logger.info(f"Found total of {len(vision_end_positions)} vision end tokens")

        image_pad_mask = seq == image_pad_token_id
        image_pad_positions = image_pad_mask.nonzero(as_tuple=True)[0]
        logger.info(f"Found total of {len(image_pad_positions)} image pad tokens")

        if len(visual_positions) == 0:
            logger.info("No visual tokens found, nothing to adjust")
            continue
        else:
            logger.info(f"Found total of {len(visual_positions)} visual tokens")
        
        # Group consecutive visual tokens into blocks (one block per image/frame)
        blocks = []
        block_start = visual_positions[0].item()
        current_token_id = int(seq[block_start].item())
        for i in range(1, len(visual_positions)):
            cur_pos = visual_positions[i].item()
            prev_pos = visual_positions[i - 1].item()
            cur_token_id = int(seq[cur_pos].item())
            if cur_pos != prev_pos + 1 or cur_token_id != current_token_id:
                blocks.append((block_start, prev_pos + 1, current_token_id))
                block_start = cur_pos
                current_token_id = cur_token_id
        blocks.append((block_start, visual_positions[-1].item() + 1, current_token_id))
        
        assert len(blocks) == len(gaussians_per_image), (
            f"Found {len(blocks)} image token blocks but {len(gaussians_per_image)} Gaussians per image specified"
        )
        
        # Build new sequence: before first block, then each adjusted block, then after last block
        new_seq_parts = []
        prev_end = 0
        
        for block_idx, (block_start, block_end, block_token_id) in enumerate(blocks):
            # Add tokens before this block
            if block_start > prev_end:
                new_seq_parts.append(seq[prev_end:block_start])
            
            # Add correct number of visual tokens for this block
            target_tokens = gaussians_per_image[block_idx]
            current_tokens = block_end - block_start
            
            if target_tokens != current_tokens:
                new_visual_tokens = torch.full(
                    (target_tokens,),
                    block_token_id,
                    dtype=seq.dtype,
                    device=seq.device
                )
                new_seq_parts.append(new_visual_tokens)
            else:
                new_seq_parts.append(seq[block_start:block_end])
            
            prev_end = block_end
        
        # Add remaining tokens after last block
        if prev_end < len(seq):
            new_seq_parts.append(seq[prev_end:])
        
        # Concatenate all parts
        new_seq = torch.cat(new_seq_parts, dim=0)
        
        input_ids = new_seq.unsqueeze(0).to(input_ids.device)
        attention_mask = torch.ones_like(input_ids)
    
    inputs["input_ids"] = input_ids
    if attention_mask is not None:
        inputs["attention_mask"] = attention_mask

    logger.info(f"final shape of input_ids: {input_ids.shape}")
    logger.info(f"final shape of attention_mask: {attention_mask.shape}")
    
    return inputs