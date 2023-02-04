# make sure you're logged in with `huggingface-cli login`
import torch
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import (
	DDIMScheduler,
	DDPMScheduler,
	#DEISMultistepScheduler,
	DPMSolverMultistepScheduler,
	DPMSolverSinglestepScheduler,
	EulerAncestralDiscreteScheduler,
	EulerDiscreteScheduler,
	FlaxDDIMScheduler,
	FlaxDDPMScheduler,
	FlaxDPMSolverMultistepScheduler,
	FlaxKarrasVeScheduler,
	FlaxLMSDiscreteScheduler,
	FlaxPNDMScheduler,
	FlaxScoreSdeVeScheduler,
	HeunDiscreteScheduler,
	IPNDMScheduler,
	KarrasVeScheduler,
	KDPM2AncestralDiscreteScheduler,
	KDPM2DiscreteScheduler,
	LMSDiscreteScheduler,
	PNDMScheduler,
	ScoreSdeVeScheduler,
	ScoreSdeVpScheduler,
)
import random
import warnings
import re
import clip
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from compel import Compel

width = 512
height = 512
model = "runwayml/stable-diffusion-v1-5"
cfg_scale = 7.5
steps = 50
sampler = "dpm_solver_s"
fp16 = True

prompt = "an astronaut fighting a dragon on mars, artstation, 8k, dramatic lighting, detailed, intricate concept art, greg rutkowski"
neg = "anime, cartoon, stippling"

pipe = StableDiffusionPipeline.from_pretrained(
	model,
	#vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse"), #this breaks with fp16
	#custom_pipeline = "waifu-research-department/long-prompt-weighting-pipeline",
	torch_dtype = torch.float16 if fp16 else torch.float32,
	safety_checker = None, #prevents loading the safety checker to save ram, disabled below anyway
)

#Converting to/from invoke-style weights is supported in numerical format only.

def inv_2_a1(prompt): return re.sub("\(([^:\)]+)\)([\d\.]+)", "(\\1:\\2)", prompt)
def a1_2_inv(prompt): return re.sub("\(([^:\)]+):([\d\.]+)\)", "(\\1)\\2", prompt)
def unweight(prompt): return re.sub("\(([^:\)]+)(:[\d\.]+)?\)[\d\.]*", "\\1", prompt)

pipe = pipe.to("mps")
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

def dummy(images, **kwargs):
    return images, False
pipe.safety_checker = dummy

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing("max")

if(sampler == "ddim"):
	pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
elif(sampler == "dpm_solver"): 
	pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
elif(sampler == "euler_a"):
	pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
elif(sampler == "euler"):
	pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
elif(sampler == "lms"):
	pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
elif(sampler == "pndm"):
	pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
elif(sampler == "dpm2"):
	pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
elif(sampler == "dpm2_a"):
	pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
elif(sampler == "dpm_solver_m"):
	pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
elif(sampler == "dpm_solver_s"):
	pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
else:
	print("invalid sampler")
	exit()

warnings.filterwarnings("ignore")
print(f"warming up...")
# First-time "warmup" pass
_ = pipe("", num_inference_steps=1, height = 64, width = 128)
warnings.filterwarnings("default")

colors = [
	'\033[31m',
	'\033[33m',
	'\033[92m',
	'\033[34m',
	'\033[35m'
]
greys = [
	'\033[38;5;8m',
	'\033[38;5;7m',
]

#handle invoke prompts and nested weights
while (prompt != inv_2_a1(prompt)):
	prompt = inv_2_a1(prompt)
while (neg != inv_2_a1(neg)):
	neg = inv_2_a1(neg)

invoke_prompt = a1_2_inv(prompt)
invoke_neg = a1_2_inv(neg)
while (invoke_prompt != a1_2_inv(invoke_prompt)):
	invoke_prompt = a1_2_inv(invoke_prompt)
while (invoke_neg != a1_2_inv(invoke_neg)):
	invoke_neg = a1_2_inv(invoke_neg)
print(f"invoke-style prompt: " + '\033[96m' + f"{invoke_prompt}\n" + '\033[91m' + f"[{invoke_neg}]" + '\033[0m')

prompt_tokens = pipe.tokenizer.tokenize(unweight(prompt))
neg_tokens = pipe.tokenizer.tokenize(unweight(neg))
print(f"prompt tokens ({len(prompt_tokens)}): ", end ="")
for i in range(min(len(prompt_tokens), 77)):
	print(colors[i % 5] + re.sub("</w>", " ", prompt_tokens[i]), end ="")
for i in range(max(len(prompt_tokens) - 77, 0)):
	print(greys[i % 2] + re.sub("</w>", " ", prompt_tokens[i + 77]), end ="")

print('\033[0m' + f"\nnegative prompt tokens ({len(neg_tokens)}): ", end ="")
for i in range(min(len(neg_tokens), 77)):
	print(colors[i % 5] + re.sub("</w>", " ", neg_tokens[i]), end ="")
for i in range(max(len(neg_tokens) - 77, 0)):
	print(greys[i % 2] + re.sub("</w>", " ", neg_tokens[i + 77]), end ="")
print('\033[0m')

while True: #keeps going until you stop it with ^c
	seed = random.randint(0, 4294967294)
	print(f"generating with seed: " + '\033[92m' + f"{seed}" + '\033[0m' + f" at {width}x{height} ({width * height / 1000000}Mpx)")
	generator = torch.Generator("cpu").manual_seed(seed)
	conditioning = compel.build_conditioning_tensor(f"{invoke_prompt}")
	neg_conditioning = compel.build_conditioning_tensor(f"{invoke_neg}")
	image = pipe(
		prompt_embeds = conditioning, 
		negative_prompt_embeds = neg_conditioning, 
		height = int(height), 
		width = int(width), 
		guidance_scale = cfg_scale, 
		num_inference_steps = steps, 
		generator=generator
	).images[0]
	md = PngInfo()
	md.add_text("a1111 Prompt", inv_2_a1(prompt))
	md.add_text("a1111 Negative prompt", inv_2_a1(neg))
	md.add_text("Invoke-style prompt", f"{invoke_prompt}\n[{invoke_neg}]")
	md.add_text("Seed", str(seed))
	md.add_text("Agent", f"diffusers_test.py")
	md.add_text("agent", f"diffusers_test.py")
	md.add_text('sd-metadata',
		'{"model": "stable diffusion", "model_weights": "'
		+ model +
		'", "model_hash": "", "app_id": "'
		+ "diffusers_test.py" +
		'", "app_version": "'
		+ "0" +
		'", "image": {"prompt": [{"prompt": "'
		+ f"{invoke_prompt} [{invoke_neg}]" +
		'", "weight": 1.0}], "steps": '
		+ str(steps) +
		', "cfg_scale": '
		+ str(cfg_scale) +
		', "threshold": 0.0, "hires_fix": false, "seamless": false, "perlin": 0.0, "width": '
		+ str(width) +
		', "height": '
		+ str(height) +
		', "seed": '
		+ str(seed) +
		', "type": "txt2img", "postprocessing": [{"type": "none"}], "sampler": "'
		+ sampler +
		'", "variations": []}}')
	image.save(f"{seed}.png", pnginfo=md)