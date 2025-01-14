from diffusers import StableDiffusionXLPipeline
import torch
from pathlib import Path

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)

guid=Path(f'SDXL_beach_results')
guid.mkdir(exist_ok=True)
pipe = pipe.to("cuda")
prompts = ['a black dog', 'a white dog', 'a running dog', 'a sitting dog',]

b_prompts =['spring', 'summer', 'autumn', 'winter']
            #'a black cat', 'a white cat', 'a running cat', 'a sitting cat']
for i in range(4):
    prompt = "a photo of a beach, "
    prompt2 = prompt + (prompts[i]+', ')
    for j in range(4):
        prompt3 = prompt2 + b_prompts[j]
        image = pipe(prompt=prompt3,
                    negative_prompt='ugly, blurry, low res, unrealistic',
                    num_images_per_prompt=10).images
        whatobj=guid/f'{prompt3}'
        whatobj.mkdir(exist_ok=True)
        for i in range(10):
            imagefile=whatobj/f'triple_{i}.png'
            image[i].save(imagefile)