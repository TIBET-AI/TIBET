from diffusers import DiffusionPipeline
import os
import torch
#from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
#pipe.enable_xformers_memory_efficient_attention()

def generate_images_batch(prompt, num_images=8):
    
    print("Generating images for prompt: ", prompt)

    if num_images < 8:
        num_batches = 1
    else:
        num_batches = num_images // 8
    images = []
    for i in range(num_batches):
        images.extend(pipe(prompt, num_images_per_prompt=num_images).images)

    return images

def generate_images(p_dict, images_dir, num_images):
    
    # Generate 10 images for original prompt, save them in images/p_dict['id']/initial/image_i.png
    # create directory for images
    if not os.path.exists(f"{images_dir}/{p_dict['id']}"):
        os.makedirs(f"{images_dir}/{p_dict['id']}/initial_prompts")

    if not os.path.exists(f"{images_dir}/{p_dict['id']}/initial_prompts/{num_images-1}.jpeg"):

        img_prompt = p_dict['initial_prompt']
        images = generate_images_batch(img_prompt, num_images=num_images)
        for i, image in enumerate(images):
            image.save(f"{images_dir}/{p_dict['id']}/initial_prompts/{i}.jpeg")
    
    else:
        print("Skipping", p_dict['id'])

    # Generate 10 images for each counterfactual prompt
    # - we consider all prompts
    for key in p_dict['result'].keys():
        for j, cf_prompt in enumerate(p_dict['result'][key]):

            # craete directory for images
            if not os.path.exists(f"{images_dir}/{p_dict['id']}/{key}/{j}"):
                os.makedirs(f"{images_dir}/{p_dict['id']}/{key}/{j}")

            if not os.path.exists(f"{images_dir}/{p_dict['id']}/{key}/{j}/{num_images-1}.jpeg"):
            
                images = generate_images_batch(cf_prompt, num_images=num_images)
                for k, image in enumerate(images):
                    image.save(f"{images_dir}/{p_dict['id']}/{key}/{j}/{k}.jpeg")

            else:
                print("Skipping", p_dict['id'], key, j)

    return p_dict


