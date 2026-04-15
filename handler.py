import os
import torch
import runpod
import base64
import io
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline, 
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    EulerDiscreteScheduler
)
import cv2
import logging
import traceback

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Caching ---
pipe = None
controlnet_canny = None
controlnet_depth = None

# Device Detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Support for Mac MPS
if torch.backends.mps.is_available():
    DEVICE = "mps"

DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

logger.info(f"Using device: {DEVICE} with dtype: {DTYPE}")

def load_models():
    global pipe, controlnet_canny, controlnet_depth
    if pipe is not None:
        return

    logger.info("Loading models into memory...")
    try:
        # 1. Load ControlNets
        logger.info(f"Loading ControlNet Canny on {DEVICE}...")
        controlnet_canny = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=DTYPE
        ).to(DEVICE)
        
        logger.info(f"Loading ControlNet Depth on {DEVICE}...")
        controlnet_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=DTYPE
        ).to(DEVICE)

        # 2. Load Base SDXL with ControlNet support
        logger.info(f"Loading SDXL Base Model on {DEVICE}...")
        # Note: Using xl-base-1.0 as the backbone for ControlNet
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=[controlnet_canny, controlnet_depth],
            torch_dtype=DTYPE,
            use_safetensors=True,
            variant="fp16" if DEVICE == "cuda" else None
        ).to(DEVICE)

        logger.info("Configuring Scheduler and Optimizations...")
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        
        # Memory optimizations for GPU
        if DEVICE == "cuda":
            pipe.enable_model_cpu_offload() # Better for VRAM if multiple ControlNets are used
            # pipe.enable_xformers_memory_efficient_attention() # Optional if xformers is installed
            
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def decode_base64_image(image_str):
    if not image_str:
        return None
    try:
        if "base64," in image_str:
            image_str = image_str.split("base64,")[1]
        image_bytes = base64.b64decode(image_str)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        return None

def process_mask(mask_str, target_size=(1024, 1024)):
    if not mask_str:
        return None
    try:
        mask_img = decode_base64_image(mask_str)
        if mask_img:
            mask_img = mask_img.resize(target_size).convert("L")
        return mask_img
    except Exception as e:
        logger.error(f"Error processing mask: {str(e)}")
        return None

def handler(job):
    try:
        # Ensure models are loaded
        load_models()
        
        job_input = job.get("input", {})
        logger.info("Received job payload. Extracting parameters...")
        
        # 1. Main Texts & Seed
        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get("negative_prompt", "")
        seed = int(job_input.get("seed", 42))
        
        # 2. Sampler 1 (Base Generation)
        steps = int(job_input.get("steps_ksampler1", 20))
        cfg = float(job_input.get("cfg_ksampler1", 7.0))
        denoise_1 = float(job_input.get("denoise_ksampler1", 1.0))
        
        # 3. Meta & Process Flags
        task_name = str(job_input.get("task", "regional_prompt"))
        job_id = job_input.get("job_id", "")
        
        logger.info(f"JOB ID: {job_id} | Task: {task_name}")
        logger.info(f"Prompt: {prompt[:50]}... | Seed: {seed}")
        
        # 4. Regional Prompts / Masks (Simplified support)
        # Note: Regional prompting logic would need specific diffusers pipeline extensions
        # but we keep the extraction logic here for compatibility.
        masks_data = []
        colors = ["yellow", "red", "green", "blue", "cyan", "magenta", "orange", "purple", "pink"]
        for color in colors:
            m_str = job_input.get(f"{color}_mask")
            p_str = job_input.get(f"{color}_prompt")
            if m_str and p_str:
                processed_mask = process_mask(m_str)
                if processed_mask:
                    masks_data.append({"mask": processed_mask, "prompt": p_str})
        
        # 5. ControlNet Parameters (Canny & Depth)
        raw_image_str = job_input.get("raw_image")
        control_images = []
        control_weights = []
        control_starts = []
        control_ends = []
        
        if raw_image_str:
            logger.info("Raw Image provided. Preparing ControlNet Inputs...")
            raw_image = decode_base64_image(raw_image_str)
            if raw_image:
                raw_image = raw_image.resize((1024, 1024))
                image_np = np.array(raw_image)
                
                # --- CONTROLNET 0: Canny ---
                canny_strength = float(job_input.get("canny_strength", 0.0))
                c_start = float(job_input.get("canny_start", 0.0))
                c_end = float(job_input.get("canny_end", 1.0))
                
                logger.info(f"Canny -> Strength: {canny_strength}")
                image_canny = cv2.Canny(image_np, 100, 200)[:, :, None]
                image_canny = np.concatenate([image_canny, image_canny, image_canny], axis=2)
                control_images.append(Image.fromarray(image_canny))
                control_weights.append(canny_strength)
                control_starts.append(c_start)
                control_ends.append(c_end)
                
                # --- CONTROLNET 1: Depth ---
                depth_strength = float(job_input.get("depth_strength", 0.0))
                d_start = float(job_input.get("depth_start", 0.0))
                d_end = float(job_input.get("depth_end", 1.0))
                
                logger.info(f"Depth -> Strength: {depth_strength}")
                control_images.append(raw_image) # Fallback depth
                control_weights.append(depth_strength)
                control_starts.append(d_start)
                control_ends.append(d_end)
                
        # If MultiControlNet was loaded but no image provided, prevent diffusers crash
        if not control_images:
            black_img = Image.new("RGB", (1024, 1024), (0, 0, 0))
            control_images = [black_img, black_img] # Must match length of pipe.controlnet
            control_weights = [0.0, 0.0]
            control_starts = [0.0, 0.0]
            control_ends = [1.0, 1.0]

        logger.info("Initializing Generator...")
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # 6. Pipeline execution
        logger.info("Starting Image Generation Process...")
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_images,
            num_inference_steps=steps,
            guidance_scale=cfg,
            controlnet_conditioning_scale=control_weights,
            control_guidance_start=control_starts,
            control_guidance_end=control_ends,
            generator=generator
        ).images[0]
        
        logger.info("Image Generated Successfully.")

        # 7. Process Output
        buffered = io.BytesIO()
        output.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "image_url": f"data:image/jpeg;base64,{img_str}",
            "job_id": job_id
        }
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR IN HANDLER: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    # Local loading and starting
    load_models()
    logger.info("Starting RunPod Serverless node...")
    runpod.serverless.start({"handler": handler})
