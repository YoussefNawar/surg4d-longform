import torch
import math
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from typing import Optional, Union
from types import MethodType

from transformers.utils import is_torchdynamo_compiling, TransformersKwargs
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast
from transformers.processing_utils import Unpack

PATCH_SIZE = 14
SPATIAL_MERGE = 2
EFFECTIVE_PATCH_SIZE = PATCH_SIZE * SPATIAL_MERGE


# original function from transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
# this is a modified version to work with raw patch features that can be used for monkey patching
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
        The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        if "custom_patch_features" in kwargs:
            image_embeds = (kwargs["custom_patch_features"],)
        else:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


def square_crop_image(image: Image.Image, xslide=0, yslide=0):
    side_length = min(image.width, image.height)
    wm, hm, rd = image.width // 2, image.height // 2, side_length // 2
    return image.crop(
        (wm - rd - xslide, hm - rd - yslide, wm + rd - xslide, hm + rd - yslide)
    )


def get_patched_qwen():
    """get a monkey patched version of Qwen2_5_VLForConditionalGeneration and Qwen2_5_VLProcesso
    that supports passing in raw patch features
    """
    model_path = "/home/tumai/models/Qwen--Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.model.forward = MethodType(forward, model.model)
    # Also patch prepare_inputs_for_generation to accept and forward custom_patch_features through generation
    orig_prepare = model.prepare_inputs_for_generation

    def prepare_inputs_for_generation_wrapper(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        custom_patch_features=None,
        **kwargs,
    ):
        model_inputs = orig_prepare(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )
        if custom_patch_features is not None:
            model_inputs["custom_patch_features"] = custom_patch_features
        return model_inputs

    model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation_wrapper, model)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor


def qwen_encode_image(image: Image.Image, model: Qwen2_5_VLForConditionalGeneration, processor: Qwen2_5_VLProcessor):
    image_inputs = processor.image_processor(  # type:ignore
        images=[image], return_tensors="pt"
    )
    pixel_values = image_inputs["pixel_values"].to(model.device).to(torch.bfloat16)
    image_grid_thw = image_inputs["image_grid_thw"].to(model.device)
    with torch.no_grad():
        feats = model.visual(pixel_values, image_grid_thw)
        return feats


def patches_to_2d(patch_features: torch.Tensor, src_image: Image.Image):
    patches_width, patches_height = (
        src_image.width // EFFECTIVE_PATCH_SIZE,
        src_image.height // EFFECTIVE_PATCH_SIZE,
    )
    return patch_features.reshape(patches_height, patches_width, -1)


def patches_to_2d_from_hw(
    patch_features: torch.Tensor, img_height: int, img_width: int
):
    patches_width, patches_height = (
        img_width // EFFECTIVE_PATCH_SIZE,
        img_height // EFFECTIVE_PATCH_SIZE,
    )
    return patch_features.reshape(patches_height, patches_width, -1)


def patch_cds(src_image: Image.Image):
    effective_patch_size = PATCH_SIZE * SPATIAL_MERGE
    patches_width, patches_height = (
        src_image.width // effective_patch_size,
        src_image.height // effective_patch_size,
    )
    patch_coords_list = []
    for i in range(patches_height):
        for j in range(patches_width):
            y1 = i * effective_patch_size
            y2 = min((i + 1) * effective_patch_size, src_image.height)
            x1 = j * effective_patch_size
            x2 = min((j + 1) * effective_patch_size, src_image.width)
            patch_coords_list.append((y1, y2, x1, x2))
    cds: torch.Tensor = torch.as_tensor(patch_coords_list)
    cds: torch.Tensor = cds.reshape(patches_height, patches_width, 2, 2)
    return cds


# This function takes in an RGB image  and a prompt
def ask_qwen_about_image(image: Image.Image, prompt: str, model: Qwen2_5_VLForConditionalGeneration, processor: Qwen2_5_VLProcessor):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a medical assistant designed to aid medical practitioners during a cholecystectomy procedure. The surgeon user will ask you a question and show you their current situation, and you give a concise answer.",
                }
            ],
        },
        # ---------------------------------------------------------------------------------------
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(  # type:ignore
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


# This function takes in the **patch features** of an image and a prompt
def ask_qwen_about_image_features(
    image_features: torch.Tensor,
    prompt: str,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: Qwen2_5_VLProcessor,
    system_prompt: str = "You are a medical assistant designed to aid medical practitioners during surgery. The surgeon user will ask you a question and show you their current situation, and you give a concise answer.",
):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(  # type:ignore
        messages, tokenize=False, add_generation_prompt=True
    )

    def closest_factor_pair(n) -> tuple[int, int]:
        root = int(math.isqrt(n))
        for a in range(root, 0, -1):
            if n % a == 0:
                return a, n // a
        raise Exception(
            "the given feature patches don't correspond to a nice rectangular size in pixels"
        )

    effective_patch_size = PATCH_SIZE * SPATIAL_MERGE
    pw_dummy, ph_dummy = closest_factor_pair(image_features.shape[0])
    inputs = processor(
        text=[text],
        images=[
            Image.new(
                "RGB",
                (effective_patch_size * pw_dummy, effective_patch_size * ph_dummy),
                color="red",
            )
        ],
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        custom_patch_features=image_features,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


def crop_patch_features(patch_feat: torch.Tensor, cw, ch, cx1, cx2, cy1, cy2):
    return patch_feat.reshape(ch, cw, -1)[cy1 : cy2 + 1, cx1 : cx2 + 1].flatten(
        end_dim=1
    )


def get_patch_segmasks(im_height, im_width):
    """generate an instance segmentation mask
    where each instance corresponds to one qwen 2.5 vl vision encoder patch"""
    raw_patches_height, raw_patches_width = (
        im_height // PATCH_SIZE,
        im_width // PATCH_SIZE,
    )
    patches_width = torch.ceil(torch.div(raw_patches_width, SPATIAL_MERGE))
    rowcol = torch.stack(
        torch.meshgrid(
            torch.arange(raw_patches_height * PATCH_SIZE),
            torch.arange(raw_patches_width * PATCH_SIZE),
            indexing="ij",
        )
    )
    patch_coords = torch.floor_divide(rowcol, EFFECTIVE_PATCH_SIZE)
    return patch_coords[0] * patches_width + patch_coords[1]


def center_crop_patch_size_multiple(img: Image.Image):
    """center crop a PIL image to a multiple of qwen 2.5 vl vision encoder patch size"""
    side_cropping_vertical = img.height % PATCH_SIZE
    side_cropping_horizontal = img.width % PATCH_SIZE
    crop_top, crop_bottom = (
        np.floor(side_cropping_vertical / 2),
        np.ceil(side_cropping_vertical / 2),
    )
    crop_left, crop_right = (
        np.floor(side_cropping_horizontal / 2),
        np.ceil(side_cropping_horizontal / 2),
    )
    return img.crop(
        (crop_left, crop_top, img.width - crop_right, img.height - crop_bottom)
    )
