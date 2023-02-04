# mps_diffusers
 A simple script for image generation using diffusers on MPS with fp16 support

```
usage: mps_diffusers.py [-h] [--profile PROFILE] -p "astronaut, mars" [-n "goofy hands"] [-c C]

AI image gen using diffusers on MPS

options:
  -h, --help            show this help message and exit
  --profile PROFILE     the profile from profiles.ini to load, omit for default
  -p "astronaut, mars"  positive prompt (in quotes)
  -n "goofy hands"      negative prompt (in quotes)
  -c C                  number of images to produce (if omitted, will generate until you terminate with ^C)
```