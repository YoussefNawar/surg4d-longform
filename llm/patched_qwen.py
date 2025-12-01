"""
Patched Qwen VL classes that support passing custom patch features directly,
bypassing the visual encoder. Uses inheritance instead of monkey patching.
"""

import torch
from typing import Optional, Union
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModel,
    Qwen2_5_VLModelOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLForConditionalGeneration,
)
from transformers.utils import is_torchdynamo_compiling, TransformersKwargs
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack


class PatchedQwen2_5_VLModel(Qwen2_5_VLModel):
    """Qwen2_5_VLModel with support for custom patch features.
    
    This allows passing pre-computed vision features instead of raw pixel values,
    which is useful when features are computed elsewhere (e.g., stored in a scene graph).
    """

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
        """
        Forward pass with optional custom_patch_features support.
        
        Additional kwargs:
            custom_patch_features: Optional pre-computed vision features to use
                instead of computing them from pixel_values.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Get custom features from kwargs (use get() to preserve for subsequent generation steps)
        custom_patch_features = kwargs.get("custom_patch_features", None)

        def _stack_features(features):
            if isinstance(features, (list, tuple)):
                return torch.cat(features, dim=0)
            return features

        if pixel_values is not None:
            if custom_patch_features is not None:
                image_embeds = custom_patch_features
            else:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = _stack_features(image_embeds).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
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
            if (
                prefill_compiled_stage or prefill_noncompiled_stage
            ) or self.rope_deltas is None:
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
                    delta = torch.zeros(
                        (batch_size, seq_length), device=inputs_embeds.device
                    )
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


class PatchedQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Qwen2_5_VLForConditionalGeneration with support for custom patch features.
    
    Overrides prepare_inputs_for_generation to properly forward custom_patch_features
    through the generation loop.
    """

    def __init__(self, config):
        super().__init__(config)
        # Swap inner model to use our patched version with custom_patch_features support
        self.model.__class__ = PatchedQwen2_5_VLModel

    def prepare_inputs_for_generation(
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
        # During decoding steps, GenerationMixin provides only the last token in input_ids
        # but the full attention_mask. Align mask length to avoid shape mismatches
        # inside Qwen2_5_VL get_rope_index when indexing with the mask.
        if attention_mask is not None and input_ids is not None:
            # If lengths differ, slice mask to align with provided input_ids tokens
            if attention_mask.shape[-1] != input_ids.shape[-1]:
                attention_mask = attention_mask[..., -input_ids.shape[-1] :]
        
        model_inputs = super().prepare_inputs_for_generation(
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


class PatchedQwen3VLModel(Qwen3VLModel):
    """Qwen3VLModel with support for custom patch features.

    This allows passing pre-computed vision features instead of raw pixel values,
    which is useful when features are computed elsewhere (e.g., stored in a scene graph).

    Custom features should be passed as:
        custom_patch_features: List of main feature tensors (one per image), each (N, hidden_dim)
        custom_deepstack_features: List of 3 tensors for deepstack injection, each (total_visual_tokens, hidden_dim)
    """

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
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        """
        Forward pass with optional custom_patch_features support.

        Additional kwargs:
            custom_patch_features: List of pre-computed main vision features (one tensor per image)
            custom_deepstack_features: List of 3 tensors for deepstack layers
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Get custom features from kwargs (use get() to preserve for subsequent generation steps)
        custom_patch_features = kwargs.get("custom_patch_features", None)
        custom_deepstack_features = kwargs.get("custom_deepstack_features", None)

        def _stack_features(features):
            if isinstance(features, (list, tuple)):
                return torch.cat(features, dim=0)
            return features

        image_mask = None
        video_mask = None
        deepstack_visual_embeds = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            if custom_patch_features is not None:
                # Use custom features instead of computing from pixels
                image_embeds = _stack_features(custom_patch_features).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                deepstack_image_embeds = custom_deepstack_features  # Can be None
            else:
                image_embeds, deepstack_image_embeds = self.get_image_features(
                    pixel_values, image_grid_thw
                )
                image_embeds = torch.cat(image_embeds, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )

            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Build visual_pos_masks and deepstack_visual_embeds
        visual_pos_masks = None
        if image_mask is not None and video_mask is not None:
            image_mask_2d = image_mask[..., 0]
            video_mask_2d = video_mask[..., 0]
            visual_pos_masks = image_mask_2d | video_mask_2d
            if deepstack_image_embeds is not None and deepstack_video_embeds is not None:
                deepstack_visual_embeds = []
                image_mask_joint = image_mask_2d[visual_pos_masks]
                video_mask_joint = video_mask_2d[visual_pos_masks]
                for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                    embed_joint = img_embed.new_zeros(
                        visual_pos_masks.sum(), img_embed.shape[-1]
                    ).to(img_embed.device)
                    embed_joint[image_mask_joint, :] = img_embed
                    embed_joint[video_mask_joint, :] = vid_embed
                    deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask_2d = image_mask[..., 0]
            visual_pos_masks = image_mask_2d
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask_2d = video_mask[..., 0]
            visual_pos_masks = video_mask_2d
            deepstack_visual_embeds = deepstack_video_embeds

        # Position ID computation (same as parent but we need to access attention_mask)
        if position_ids is None:
            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask.get("full_attention")
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = (
                        attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    )
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

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
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class PatchedQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3VLForConditionalGeneration with support for custom patch features.

    Overrides prepare_inputs_for_generation to properly forward custom_patch_features
    through the generation loop.
    """

    def __init__(self, config):
        super().__init__(config)
        # Swap inner model to use our patched version with custom_patch_features support
        self.model.__class__ = PatchedQwen3VLModel

    def prepare_inputs_for_generation(
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
        custom_patch_features=None,
        custom_deepstack_features=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
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
            **kwargs,
        )
        if custom_patch_features is not None:
            model_inputs["custom_patch_features"] = custom_patch_features
        if custom_deepstack_features is not None:
            model_inputs["custom_deepstack_features"] = custom_deepstack_features
        return model_inputs

