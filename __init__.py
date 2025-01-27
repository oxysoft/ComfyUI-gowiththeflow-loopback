import math
import torch
import numpy as np
import cv2
from .noise_warper import NoiseWarper
import comfy.samplers
import comfy.sample
import latent_preview
import comfy.utils

###############################################################################
# bgr_to_flow function from your code snippet
###############################################################################
def bgr_to_flow(bgr_image, max_flow=20.0):
    """
    Convert a BGR image back to a flow map (H,W,2).
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    angle = hsv[..., 0] * np.pi / 180.0 * 2.0
    magnitude = hsv[..., 2] / 255.0
    flow_map = np.zeros((hsv.shape[0], hsv.shape[1], 2), dtype=np.float32)
    flow_map[..., 0], flow_map[..., 1] = cv2.polarToCart(magnitude, angle)
    # scale
    flow_map *= max_flow
    return flow_map

########################################
# GLOBAL dict of warpers 
########################################
WARPERS = {}

########################################
# ComfyUI Node
########################################
class NoiseWarperNode:
    """
    Node that:
      - Accepts an RGB flow image 
      - Decodes it to (dx, dy)
      - Persists a NoiseWarper
      - Outputs a LATENT [4, H//8, W//8] by default
      - If width/height=0, we use the flow image's size
      - Automatically picks a random seed each time we re-init
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_image": ("IMAGE",),
                "width": ("INT", {"default":0, "min":0, "max":4096, "step":1}),
                "height": ("INT", {"default":0, "min":0, "max":4096, "step":1}),
                # "reset": ("BOOL", {"default":False}),
                "noise_channels": ("INT", {"default":4, "min":1, "max":8}),
                "scale_factor": ("INT", {"default":1, "min":1, "max":8}),
                "noise_scale": ("FLOAT", {"default":1.0, "min":0.0, "max":10.0, "step":0.01, "tooltip": "The amount of noise added to the latent image."}),
            },
        }

    RETURN_TYPES = ("LATENT","IMAGE")
    FUNCTION = "warp_noise"
    CATEGORY = "latent"
    
    @classmethod
    def IS_CHANGED(flow_image, width, height, noise_channels, scale_factor, noise_scale):
        return float("NaN")

    def warp_noise(self, flow_image, width, height, noise_channels, scale_factor, noise_scale):
        # Ensure we only have a single flow image (squeeze batch dimension if needed)
        if len(flow_image.shape) == 4:  # [B,H,W,C]
            assert flow_image.shape[0] == 1, "Only single flow image input is supported"
            flow_image = flow_image.squeeze(0)  # Remove batch dimension -> [H,W,C]
        else:
            assert len(flow_image.shape) == 3, "Flow image must have 3 or 4 dimensions"

        node_id = id(self)
        flow_h, flow_w, c = flow_image.shape
        print(c, flow_h, flow_w)
        assert c == 3, "Flow image must have 3 channels (RGB)"

        # 1) If width=0 or height=0, use the flow image's shape
        if width == 0:
            width = flow_w // 8
        if height == 0:
            height = flow_h // 8

        # 2) Convert ComfyUI tensor -> BGR [uint8, 0..255]
        flow_bgr = (flow_image.cpu().numpy() * 255).astype(np.uint8)
        # 3) bgr_to_flow => shape [H, W, 2]
        flow_map = bgr_to_flow(flow_bgr, max_flow=20.0)
        dx_np = flow_map[..., 0]  # shape [H, W]
        dy_np = flow_map[..., 1]

        # to torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dx = torch.from_numpy(dx_np).float().to(device)
        dy = torch.from_numpy(dy_np).float().to(device)

        # 4) Check if we have a warper or need to re-init
        warper = WARPERS.get(node_id, None)
        do_reset = (warper is None) \
                   or (warper.h != height) or (warper.w != width) \
                   or (warper.c != noise_channels)

        if do_reset:
            # Use a random seed each time we re-init
            random_seed = np.random.randint(0, (1 << 31) - 1)
            np.random.seed(random_seed & 0xFFFFFFFF)
            torch.manual_seed(random_seed)

            warper = NoiseWarper(
                c=noise_channels,
                h=height,
                w=width,
                device=device,
                scale_factor=scale_factor
            )
            WARPERS[node_id] = warper


        # 5) Warp
        random_seed = np.random.randint(0, (1 << 31) - 1)
        np.random.seed(random_seed & 0xFFFFFFFF)
        torch.manual_seed(random_seed)
        warper(dx, dy)

        # 6) Retrieve noise => [noise_channels, H, W]
        noise = warper.noise

        # 7) Convert to a 4-channel latent, shape [4, H//8, W//8]
        # If noise_channels <4 => pad. If >4 => slice.
        latent_channels = 4
        if noise_channels < latent_channels:
            pad_c = latent_channels - noise_channels
            pad_noise = torch.randn((pad_c, height, width), device=device)
            final_noise = torch.cat([noise, pad_noise], dim=0)
        else:
            final_noise = noise[:latent_channels]

        # downscale to [4, H//8, W//8]
        latent_h, latent_w = height, width
        latent = final_noise * noise_scale
        # latent = torch.nn.functional.interpolate(
        #     final_noise.unsqueeze(0),
        #     size=(latent_h, latent_w),
        #     mode="linear"
        # )[0]

        print("RETURNING LATENT NOISE: ", latent.shape)
        # Wrap the latent in a dictionary with "samples" key to match SDXL format
        # Normalize noise from [-4,4] range to [0,1] range
        normalized_noise = (noise + abs(noise.min())) / (noise.max() - noise.min())
        return ({"samples": latent.unsqueeze(0)}, normalized_noise.permute(1, 2, 0).unsqueeze(0),)

class KSamplerNoiseless:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "disable_noise": ("BOOLEAN", {"default": True, "tooltip": "If True, the noise will not be added to the latent image."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = "sampling"
    DESCRIPTION = "Uses the provided model, positive and negative conditioning to denoise the latent image."

    def sample(self, model, disable_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise)

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    generator = torch.manual_seed(seed)
    baseline_noise = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")

    # Scale latent_image to match baseline_noise range
    baseline_min = baseline_noise.min()
    baseline_max = baseline_noise.max()
    latent_min = latent_image.min() 
    latent_max = latent_image.max()

    # Apply min-max normalization to match baseline noise range
    # latent_image = (latent_image - latent_min) / (latent_max - latent_min) 
    # latent_image = latent_image * (baseline_max - baseline_min) + baseline_min

    # if not disable_noise:
    #     batch_inds = latent["batch_index"] if "batch_index" in latent else None
    #     latent_image = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    # noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    # print("Noise range: ", latent_image.min(), latent_image.max())


    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, latent_image, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=True, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )



# Tell ComfyUI about our node
NODE_CLASS_MAPPINGS = {
    "NoiseWarperNode": NoiseWarperNode,
    "KSamplerNoiseless": KSamplerNoiseless
}