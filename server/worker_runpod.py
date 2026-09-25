import os, shutil, requests, random, time, uuid, boto3, runpod
from pathlib import Path
from urllib.parse import urlsplit
from datetime import datetime

import torch
import numpy as np
from PIL import Image
import folder_paths

def download_file(url, save_dir, file_name, overwrite=True):
    os.makedirs(save_dir, exist_ok=True)
    file_suffix = os.path.splitext(urlsplit(url).path)[1]
    file_name_with_suffix = file_name + file_suffix
    file_path = os.path.join(save_dir, file_name_with_suffix)
    if os.path.exists(file_path) and not overwrite:
        return file_path
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

def _opt_flag(v, default=True):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)

import comfy
from nodes import NODE_CLASS_MAPPINGS
import nodes as nodes_module

from comfy_execution.utils import get_executing_context
if get_executing_context() is None:
    from types import SimpleNamespace
    import comfy_execution.utils
    comfy_execution.utils.get_executing_context = lambda: SimpleNamespace(node_id="notebook_manual_run")

import asyncio
from server import PromptServer

# Headless stub: core nodes don't register routes, but keep parity with
# TostUpscale3 worker so future custom nodes don't break headless import.
class _NoopRoutes:
    def _decorator(self, *args, **kwargs):
        def wrap(fn):
            return fn
        return wrap
    def __getattr__(self, name):
        return self._decorator

class _DummyPromptServer:
    routes = _NoopRoutes()
    def send_sync(self, *args, **kwargs):
        pass
    def add_on_prompt_handler(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

if getattr(PromptServer, 'instance', None) is None:
    PromptServer.instance = _DummyPromptServer()

# Core new-API nodes (TextEncodeQwenImage21, MiniMaxH3ImageToVideo,
# MarigoldV2PostProcess, ResolutionSelector, ...) live in comfy_extras/
# and only register here. No custom nodes in this workflow.
asyncio.run(nodes_module.init_builtin_extra_nodes())

# --- old-API (instantiated) ---
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
LoraLoaderModelOnly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
ConditioningZeroOut = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()

# --- new-API (classmethod .execute) ---
TextEncodeQwenImage21 = NODE_CLASS_MAPPINGS["TextEncodeQwenImage21"]
QwenImage21Cache = NODE_CLASS_MAPPINGS["QwenImage21Cache"]
MiniMaxH3ImageToVideo = NODE_CLASS_MAPPINGS["MiniMaxH3ImageToVideo"]
ConditioningLoader = NODE_CLASS_MAPPINGS["ConditioningLoader"]
MarigoldV2PostProcess = NODE_CLASS_MAPPINGS["MarigoldV2PostProcess"]
ResolutionSelector = NODE_CLASS_MAPPINGS["ResolutionSelector"]
BasicScheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]
BasicGuider = NODE_CLASS_MAPPINGS["BasicGuider"]
KSamplerSelect = NODE_CLASS_MAPPINGS["KSamplerSelect"]
RandomNoise = NODE_CLASS_MAPPINGS["RandomNoise"]
DisableNoise = NODE_CLASS_MAPPINGS["DisableNoise"]
ManualSigmas = NODE_CLASS_MAPPINGS["ManualSigmas"]
SamplerCustomAdvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]
VAEDecodeAudio = NODE_CLASS_MAPPINGS["VAEDecodeAudio"]
CreateVideo = NODE_CLASS_MAPPINGS["CreateVideo"]

from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow as _AuraCls
ModelSamplingAuraFlow = _AuraCls()
from comfy_extras.nodes_logic import SwitchNode as ComfySwitchNode
from comfy_extras.nodes_math import MathExpressionNode as ComfyMathExpression

with torch.inference_mode():
    # T2I (Krea-2 Turbo)
    unet_t2i = UNETLoader.load_unet("krea2_turbo_fp8_scaled.safetensors", "default")[0]
    clip_t2i = CLIPLoader.load_clip("qwen3vl_4b_fp8_scaled.safetensors", type="krea2")[0]
    vae_t2i = VAELoader.load_vae("qwen_image_vae.safetensors")[0]
    # Albedo (Marigold V2)
    unet_albedo = UNETLoader.load_unet("qwen_image_edit_2509_int8_convrot.safetensors", "default")[0]
    vae_albedo = VAELoader.load_vae("marigold_v2_albedo_vae.safetensors")[0]
    cond_albedo = ConditioningLoader.execute("marigold_v2_albedo_conditioning.safetensors")[0]
    # Qwen Image 2.1 Edit
    unet_qwen = UNETLoader.load_unet("qwen_image_2.1_int8_convrot.safetensors", "default")[0]
    clip_qwen = CLIPLoader.load_clip("qwen3vl_8b_int8_convrot.safetensors", type="qwen_image")[0]
    vae_qwen = VAELoader.load_vae("qwen_image_2.1_vae_bf16.safetensors")[0]
    # MiniMax H3 I2V
    unet_h3 = UNETLoader.load_unet("minimax_h3_fl2va_pruned_int8_convrot.safetensors", "default")[0]
    clip_h3 = CLIPLoader.load_clip("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax")[0]
    vae_h3_video = VAELoader.load_vae("minimax_h3_video_vae_fp16.safetensors")[0]
    vae_h3_audio = VAELoader.load_vae("minimax_h3_audio_vae_fp32.safetensors")[0]

@torch.inference_mode()
def generate(input):
    try:
        tmp_dir = "/content/ComfyUI/output"
        os.makedirs(tmp_dir, exist_ok=True)
        unique_id = uuid.uuid4().hex[:6]
        current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        s3_access_key_id = os.getenv('s3_access_key_id')
        s3_secret_access_key = os.getenv('s3_secret_access_key')
        s3_endpoint_url = os.getenv('s3_endpoint_url')
        s3_region_name = os.getenv('s3_region_name')
        s3_bucket_name = os.getenv('s3_bucket_name')
        s3_bucket_folder = os.getenv('s3_bucket_folder')
        s3 = boto3.client('s3', aws_access_key_id=s3_access_key_id, aws_secret_access_key=s3_secret_access_key, endpoint_url=s3_endpoint_url, region_name=s3_region_name)

        values = input["input"]
        job_id = values['job_id']

        # ============ workflow defaults (from text_image_albedo_green_screen_video.json) ============
        # --- T2I ---
        prompt_t2i = values.get('prompt_t2i', "pixel art, 16-bit pixel art, full body a furious female warrior, walking, eyes blazing, lips,fists clenched, grips a bloodied axe, her tattered trench coat flares as she lunges at a tenacious, multi-eyed eldritch horror, motion-blurred limbs, chiaroscuro highlights, 8K hyperreal grit, shallow depth-of-field, cinematic recoil dynamics, ultra-detailed sprite-icon textures, emotional fury collides with primal jealousy raw aggression erupts in 90fps kinetic chaos, thunderous impact frame, airborne shrapnel, cracked cobblestones, breathing fireflies in shadow")
        t2i_lora = values.get('t2i_lora', 'k2-pixel128.safetensors')
        t2i_lora_strength = values.get('t2i_lora_strength', 1.0)
        t2i_steps = values.get('t2i_steps', 8)
        t2i_cfg = values.get('t2i_cfg', 1.0)
        t2i_sampler = values.get('t2i_sampler', 'euler')
        t2i_scheduler = values.get('t2i_scheduler', 'simple')
        # fixed second LoRA from subgraph 547/51
        bypass_lora = values.get('bypass_lora', 'krea2filterbypass3.safetensors')
        bypass_strength = values.get('bypass_strength', 100.0)
        t2i_aspect = values.get('t2i_aspect', '1:1 (Square)')
        t2i_megapixels = values.get('t2i_megapixels', 1.0)
        # --- Albedo ---
        enable_albedo = _opt_flag(values.get('enable_albedo', True), True)
        albedo_sampler = values.get('albedo_sampler', 'euler')
        albedo_sigmas = values.get('albedo_sigmas', '0.5, 0')
        albedo_prediction = values.get('albedo_prediction', 'albedo')
        albedo_shift = values.get('albedo_shift', 1.73)
        albedo_sampling = values.get('albedo_sampling', 'img_to_img_velocity')
        # --- Qwen Edit ---
        enable_qwen_edit = _opt_flag(values.get('enable_qwen_edit', True), True)
        prompt_edit = values.get('prompt_edit', "16-bit pixel art full body character, keep the character, remove ground and make <image2> background of <image1> ")
        negative_edit = values.get('negative_edit', "")
        edit_steps = values.get('edit_steps', 25)
        edit_cfg = values.get('edit_cfg', 1.0)
        edit_sampler = values.get('edit_sampler', 'euler')
        edit_scheduler = values.get('edit_scheduler', 'simple')
        edit_resolution = values.get('edit_resolution', 0)
        cache_device = values.get('cache_device', 'auto')
        cache_dtype = values.get('cache_dtype', 'default')
        background_image = values.get('background_image')  # optional URL; None -> solid green
        input_image = values.get('input_image')  # optional URL; if set, skip T2I and use it directly
        # --- MiniMax H3 ---
        prompt_video = values.get('prompt_video', "Front view, full body of the exact same character as the reference image, facing the camera. Entire body visible from head to feet with clear margin around the silhouette. Character performs a smooth loopable walk cycle in place: alternating mid-stride poses, one foot stepping forward while the opposite foot pushes off, arms swinging naturally in opposition, torso upright, head facing forward at all times. Character stays centered in frame at constant scale: no walking toward the camera, no turning, no jumping, no dancing, no extra limbs, no morphing. Fixed static camera: no movement, no zoom, no pan, no tilt, no gradients, no shadows, no texture, no lighting variation, identical in every frame. Preserve the reference art style, pixel-art look, colors, lighting and proportions exactly; do not redesign the character. Consistent character identity across all frames. Audio: soft rhythmic footsteps only, no music, no background noise.")
        video_width = values.get('video_width')  # None -> ResolutionSelector 0.4MP
        video_height = values.get('video_height')
        video_aspect = values.get('video_aspect', '1:1 (Square)')
        video_megapixels = values.get('video_megapixels', 0.4)
        duration = values.get('duration', 5.0)
        turbo_mode = values.get('turbo_mode', True)
        turbo_lora = values.get('turbo_lora', 'minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors')
        turbo_strength = values.get('turbo_strength', 1.0)
        turbo_steps = values.get('turbo_steps', 3)
        full_steps = values.get('full_steps', 20)
        enable_loop_lora = values.get('enable_loop_lora', False)  # subgraph 556 switch, default false
        loop_lora = values.get('loop_lora', 'minimax_h3_looping_sketch_anime_v1.safetensors')
        loop_strength = values.get('loop_strength', 1.0)
        fps = values.get('fps', 24)
        seed = values.get('seed', 0)

        if seed == 0:
            random.seed(int(time.time()))
            seed = random.randint(0, 18446744073709551615)

        # --- resolutions (ResolutionSelector 548 + 554) ---
        w_t2i, h_t2i = ResolutionSelector.execute(t2i_aspect, t2i_megapixels, 32, preview=None)[:2]
        w_t2i, h_t2i = int(w_t2i), int(h_t2i)
        if video_width is None or video_height is None:
            w_vid, h_vid = ResolutionSelector.execute(video_aspect, video_megapixels, 32, preview=None)[:2]
            w_vid, h_vid = int(w_vid), int(h_vid)
        else:
            w_vid, h_vid = int(video_width), int(video_height)

        # ============ 1. T2I (subgraph 547: 544->545->51->533->546->534) ============
        # Skipped when input_image is provided.
        input_image_path = None
        if input_image:
            input_image_path = download_file(url=input_image, save_dir=folder_paths.get_input_directory(),
                                             file_name=f'input_image_{unique_id}')
            image_t2i = LoadImage.load_image(input_image_path)[0]
        else:
            model_t2i = LoraLoaderModelOnly.load_lora_model_only(unet_t2i, t2i_lora, strength_model=t2i_lora_strength)[0]
            model_t2i = LoraLoaderModelOnly.load_lora_model_only(model_t2i, bypass_lora, strength_model=bypass_strength)[0]
            positive_t2i = CLIPTextEncode.encode(clip_t2i, prompt_t2i)[0]
            negative_t2i = ConditioningZeroOut.zero_out(positive_t2i)[0]
            latent_t2i = EmptyLatentImage.generate(width=w_t2i, height=h_t2i, batch_size=1)[0]
            latent_t2i = KSampler.sample(model_t2i, seed, t2i_steps, t2i_cfg, t2i_sampler, t2i_scheduler,
                                         positive_t2i, negative_t2i, latent_t2i, denoise=1.0)[0]
            comfy.model_management.unload_all_models()
            image_t2i = VAEDecode.decode(vae_t2i, latent_t2i)[0].detach()

        # ============ 2. Albedo (subgraph 530) ============
        # Skipped when enable_albedo=false -> passthrough image_t2i.
        if enable_albedo:
            model_albedo = LoraLoaderModelOnly.load_lora_model_only(unet_albedo, "marigold_v2_albedo.safetensors", strength_model=1.0)[0]
            model_albedo = ModelSamplingAuraFlow.patch_aura(model_albedo, albedo_shift, sampling=albedo_sampling)[0]
            guider_albedo = BasicGuider.execute(model_albedo, cond_albedo)[0]
            noise_albedo = DisableNoise.execute()[0]
            sampler_albedo = KSamplerSelect.execute(albedo_sampler)[0]
            sigmas_albedo = ManualSigmas.execute(albedo_sigmas)[0]
            latent_albedo = VAEEncode.encode(pixels=image_t2i, vae=vae_albedo)[0]
            latent_albedo = SamplerCustomAdvanced.execute(noise_albedo, guider_albedo, sampler_albedo, sigmas_albedo, latent_albedo)[0]
            comfy.model_management.unload_all_models()
            decoded_albedo = VAEDecode.decode(vae_albedo, latent_albedo)[0].detach()
            image_albedo = MarigoldV2PostProcess.execute(decoded_albedo, albedo_prediction)[0]
        else:
            image_albedo = image_t2i

        # ============ 3+4. Background + Qwen Edit (subgraph 459) ============
        # Skipped when enable_qwen_edit=false -> passthrough image_albedo.
        bg_path = None
        if enable_qwen_edit:
            if background_image:
                # LoadImage only accepts paths under the input dir (path-traversal guard)
                bg_path = download_file(url=background_image, save_dir=folder_paths.get_input_directory(),
                                        file_name=f'background_{unique_id}')
                image_bg = LoadImage.load_image(bg_path)[0]
            else:
                # default green screen (matches repo title); Qwen Edit prompt composites it as <image2>
                image_bg = torch.zeros((1, 1024, 1024, 3), dtype=image_albedo.dtype, device=image_albedo.device)
                image_bg[..., 1] = 1.0

            model_edit = QwenImage21Cache.execute(unet_qwen, cache_device, cache_dtype)[0]
            positive_edit, negative_edit_c, latent_empty = TextEncodeQwenImage21.execute(
                clip_qwen, prompt_edit, negative_edit, vae_qwen, edit_resolution,
                images={"image_1": image_albedo, "image_2": image_bg})
            latent_empty_sized = EmptyLatentImage.generate(width=w_t2i, height=h_t2i, batch_size=1)[0]
            # ComfySwitchNode 468: switch=true -> EmptyLatent (custom size), false -> edit latent
            latent_edit = ComfySwitchNode.execute(True, latent_empty_sized, latent_empty)[0]
            latent_edit = KSampler.sample(model_edit, seed, edit_steps, edit_cfg, edit_sampler, edit_scheduler,
                                          positive_edit, negative_edit_c, latent_edit, denoise=1.0)[0]
            comfy.model_management.unload_all_models()
            image_edit = VAEDecode.decode(vae_qwen, latent_edit)[0].detach()
        else:
            image_edit = image_albedo

        # ============ 5. MiniMax H3 I2V (subgraph 483) ============
        # duration -> length grid: max(5, round(a*24)) snapped to 17k+5
        length = ComfyMathExpression.execute(
            "max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17",
            values={"a": float(duration)})[1]
        length = int(length)
        steps_h3 = int(turbo_steps) if bool(turbo_mode) else int(full_steps)
        model_h3 = LoraLoaderModelOnly.load_lora_model_only(unet_h3, turbo_lora, strength_model=float(turbo_strength))[0] \
            if bool(turbo_mode) else unet_h3
        # ComfySwitchNode 556: loop LoRA disabled by default
        if bool(enable_loop_lora):
            model_h3 = LoraLoaderModelOnly.load_lora_model_only(model_h3, loop_lora, strength_model=float(loop_strength))[0]
        positive_h3, latent_h3 = MiniMaxH3ImageToVideo.execute(
            clip_h3, vae_h3_video, prompt_video, w_vid, h_vid, length,
            first_frame=image_edit, last_frame=None)
        guider_h3 = BasicGuider.execute(model_h3, positive_h3)[0]
        sampler_h3 = KSamplerSelect.execute("res_multistep")[0]
        noise_h3 = RandomNoise.execute(seed)[0]
        sigmas_h3 = BasicScheduler.execute(model_h3, "simple", steps_h3, 1.0)[0]
        latent_h3 = SamplerCustomAdvanced.execute(noise_h3, guider_h3, sampler_h3, sigmas_h3, latent_h3)[0]
        comfy.model_management.unload_all_models()
        frames_h3 = VAEDecode.decode(vae_h3_video, latent_h3)[0].detach()
        audio_h3 = VAEDecodeAudio.execute(vae_h3_audio, latent_h3)[0]
        comfy.model_management.unload_all_models()
        video = CreateVideo.execute(frames_h3, float(fps), audio_h3, "auto", "sRGB", "none")[0]
        out_path = f"{tmp_dir}/tost_sprite_studio.mp4"
        video.save_to(out_path)
        comfy.model_management.unload_all_models()

        s3_key = f"{s3_bucket_folder}/tost_sprite_studio-{current_time}-{seed}-{unique_id}.mp4"
        s3.upload_file(out_path, s3_bucket_name, s3_key, ExtraArgs={'ContentType': 'video/mp4'})
        result_url = f"{s3_endpoint_url}/{s3_bucket_name}/{s3_key}"

        return {"job_id": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"job_id": job_id if 'job_id' in locals() else None, "result": str(e), "status": "FAILED"}
    finally:
        for p in [locals().get('bg_path'), locals().get('input_image_path')]:
            if p and os.path.exists(p):
                os.remove(p)
        directory_path = Path(tmp_dir)
        if directory_path.exists():
            shutil.rmtree(directory_path)
            print(f"Directory {directory_path} has been removed successfully.")
        else:
            print(f"Directory {directory_path} does not exist.")

runpod.serverless.start({"handler": generate})
