import torch

def text_editing(model, source_image, style_image, style_text, target_text,
                   starting_layer=10, ddim_steps=50, scale=2):

    device = model.device
    with torch.no_grad():
        prompts = [style_text, target_text]
        inversion_prompt = [style_text]
        start_code, latents_list = model.inversion(source_image,
                                                style_image,
                                                inversion_prompt,
                                                guidance_scale=scale,
                                                num_inference_steps=ddim_steps,
                                                return_intermediates=True)
        start_code = start_code.expand(len(prompts), -1, -1, -1)
        image_GaMuSA = model(
                style_image.expand(len(prompts), -1, -1, -1),
                prompts,
                num_inference_steps=ddim_steps,
                latents=start_code,
                guidance_scale=scale,
                enable_GaMuSA=True
        )
        image_GaMuSA = image_GaMuSA.clamp(0, 1)
        image_GaMuSA = image_GaMuSA.cpu().permute(0, 2, 3, 1).numpy()

    return [
        image_GaMuSA[0],
        image_GaMuSA[1],
    ] 
