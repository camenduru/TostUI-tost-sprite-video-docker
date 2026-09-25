FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PYTHONDONTWRITEBYTECODE=True
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libegl1 libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev cmake && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg xvfb && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda_12.9.1_575.57.08_linux.run -d /content -o cuda_12.9.1_575.57.08_linux.run && sh cuda_12.9.1_575.57.08_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    rm -f /content/cuda_12.9.1_575.57.08_linux.run

USER camenduru

RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129 && \
    pip install --no-cache-dir xformers --extra-index-url https://download.pytorch.org/whl/cu129 && \
    git clone --depth 1 --branch master https://github.com/comfyanonymous/ComfyUI /content/ComfyUI && cd /content/ComfyUI && pip install -r requirements.txt && \
    pip install --no-cache-dir opencv-python imageio imageio-ffmpeg ffmpeg-python moviepy timm scikit-image matplotlib diffusers accelerate peft einops color-matcher runpod boto3 && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors -d /content/ComfyUI/models/diffusion_models -o krea2_turbo_fp8_scaled.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/marigold-v2-0/resolve/main/diffusion_models/qwen_image_edit_2509_int8_convrot.safetensors -d /content/ComfyUI/models/diffusion_models -o qwen_image_edit_2509_int8_convrot.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/diffusion_models/qwen_image_2.1_int8_convrot.safetensors -d /content/ComfyUI/models/diffusion_models -o qwen_image_2.1_int8_convrot.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors -d /content/ComfyUI/models/diffusion_models -o minimax_h3_fl2va_pruned_int8_convrot.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors -d /content/ComfyUI/models/text_encoders -o qwen3vl_4b_fp8_scaled.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors -d /content/ComfyUI/models/text_encoders -o qwen3vl_8b_int8_convrot.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors -d /content/ComfyUI/models/text_encoders -o qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors -d /content/ComfyUI/models/vae -o qwen_image_vae.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/marigold-v2-0/resolve/main/vae/marigold_v2_albedo_vae.safetensors -d /content/ComfyUI/models/vae -o marigold_v2_albedo_vae.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors -d /content/ComfyUI/models/vae -o qwen_image_2.1_vae_bf16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors -d /content/ComfyUI/models/vae -o minimax_h3_video_vae_fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors -d /content/ComfyUI/models/vae -o minimax_h3_audio_vae_fp32.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/e-n-v-y/Krea-2-Pixel-Art/resolve/main/k2-pixel128.safetensors -d /content/ComfyUI/models/loras -o k2-pixel128.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/Krea-2/resolve/main/loras/krea2_darkbrush.safetensors -d /content/ComfyUI/models/loras -o krea2_darkbrush.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/uzumix/krea2filterbypass3.safetensors/resolve/main/krea2filterbypass3.safetensors -d /content/ComfyUI/models/loras -o krea2filterbypass3.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/marigold-v2-0/resolve/main/loras/marigold_v2_albedo.safetensors -d /content/ComfyUI/models/loras -o marigold_v2_albedo.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Kijai/MiniMax-H3_comfy/resolve/main/loras/minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors -d /content/ComfyUI/models/loras -o minimax_h3_taomate_3step_lora_avg_rank_19_bf16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime/resolve/main/minimax_h3_looping_sketch_anime_v1.safetensors -d /content/ComfyUI/models/loras -o minimax_h3_looping_sketch_anime_v1.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/Comfy-Org/marigold-v2-0/resolve/main/embeddings/marigold_v2_albedo_conditioning.safetensors -d /content/ComfyUI/models/embeddings -o marigold_v2_albedo_conditioning.safetensors

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py
