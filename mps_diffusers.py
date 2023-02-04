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
from configparser import ConfigParser
import argparse
import json
import random
import warnings
import re
import clip
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from compel import Compel

parser = argparse.ArgumentParser(description="AI image gen using diffusers on MPS")
parser.add_argument('--profile', type=str, help="the profile from profiles.ini to load, omit for default")
parser.add_argument('-p', metavar='"astronaut, mars"', type=str, help="positive prompt (in quotes)", required=True)
parser.add_argument('-n', metavar='"goofy hands"', type=str, help="negative prompt (in quotes)")
args = parser.parse_args()

conf = ConfigParser()
conf.read('profiles.ini')

if args.profile and args.profile in conf.sections():
	choice = args.profile
else:
	choice = False

# copy our profile from the .ini, will read from this copy below
st = dict(conf[choice]) if choice else dict(conf['DEFAULT'])

# this is a hack to cast to type without hardcoding getters
for k,v in st.items():
    if v[0].isdigit():
        try:
            st[k] = int(v)
        except ValueError:
            st[k] = float(v)

# this is because the above hack doesn't cover bools
st['fp16'] = json.loads(st['fp16'].lower())

prompt = args.p
neg = args.n or ""

pipe = StableDiffusionPipeline.from_pretrained(
	st['model'],
	#vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse"), #this breaks with fp16
	#custom_pipeline = "waifu-research-department/long-prompt-weighting-pipeline",
	torch_dtype = torch.float16 if st['fp16'] else torch.float32,
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

sampler_dict = {"ddim":       DDIMScheduler.from_config(pipe.scheduler.config),
				"dpm_solver": DDIMScheduler.from_config(pipe.scheduler.config),
				"euler_a": EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
				"euler":   EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
				"lms":  LMSDiscreteScheduler.from_config(pipe.scheduler.config),
				"pndm": PNDMScheduler.from_config(pipe.scheduler.config),
				"dpm2": KDPM2DiscreteScheduler.from_config(pipe.scheduler.config),
				"dpm2_a": KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config),
				"dpm_solver_m": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config),
				"dpm_solver_s": DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config)
				}

try:
	pipe.scheduler = sampler_dict[st['sampler']]
except KeyError:
	print(f"{st['sampler']} - invalid sampler")
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
	print(f"generating with seed: " + '\033[92m' + f"{seed}" + '\033[0m' + f" at {st['width']}x{st['height']} ({st['width'] * st['height'] / 1000000}Mpx)")
	generator = torch.Generator("cpu").manual_seed(seed)
	conditioning = compel.build_conditioning_tensor(f"{invoke_prompt}")
	neg_conditioning = compel.build_conditioning_tensor(f"{invoke_neg}")
	image = pipe(
		prompt_embeds = conditioning, 
		negative_prompt_embeds = neg_conditioning, 
		height = st['height'], 
		width = st['width'], 
		guidance_scale = st['cfg_scale'], 
		num_inference_steps = st['steps'], 
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
		+ st['model'] +
		'", "model_hash": "", "app_id": "'
		+ "diffusers_test.py" +
		'", "app_version": "'
		+ "0" +
		'", "image": {"prompt": [{"prompt": "'
		+ f"{invoke_prompt} [{invoke_neg}]" +
		'", "weight": 1.0}], "steps": '
		+ str(st['steps']) +
		', "cfg_scale": '
		+ str(st['cfg_scale']) +
		', "threshold": 0.0, "hires_fix": false, "seamless": false, "perlin": 0.0, "width": '
		+ str(st['width']) +
		', "height": '
		+ str(st['height']) +
		', "seed": '
		+ str(seed) +
		', "type": "txt2img", "postprocessing": [{"type": "none"}], "sampler": "'
		+ st['sampler'] +
		'", "variations": []}}')
	image.save(f"{seed}.png", pnginfo=md)