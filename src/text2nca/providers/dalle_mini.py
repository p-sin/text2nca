from text2nca.providers.provider import Provider
# credit: https://github.com/borisdayma/dalle-mini/blob/main/tools/inference/inference_pipeline.ipynb

from PIL import Image

import random
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key
from tqdm.notebook import trange

DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"
#DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest" 
DALLE_COMMIT_ID = None

VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

# compile and parallelize model function to use multiple devices
# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

class DallEMini(Provider):
    '''
    Interface for the providers that turns a text string into an image to be processed.
    '''

    def __init__(self):
        global model
        #global params
        global vqgan
        #global vqgan_params
        # Load dalle-mini
        model, self.params = DalleBart.from_pretrained(
            DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
        )

        # Load VQGAN
        vqgan, self.vqgan_params = VQModel.from_pretrained(
            VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
        )
        
        # replicate model parameters on each device for faster inference
        self.params = replicate(self.params)
        self.vqgan_params = replicate(self.vqgan_params)


        # create a random key
        seed = random.randint(0, 2**32 - 1)
        self.key = jax.random.PRNGKey(seed)

        self.processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)


        

    def get_image(self, prompt):
        '''
        Generate an image based on a prompt
        '''

        tokenized_prompt = replicate(self.processor([prompt]))

        # number of predictions per prompt
        n_predictions = 8

        # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
        gen_top_k = None
        gen_top_p = None
        temperature = None
        cond_scale = 10.0


        # generate image
        images = []
        for i in trange(max(n_predictions // jax.device_count(), 1)):
            # get a new key
            self.key, subkey = jax.random.split(self.key)
            # generate images
            encoded_images = p_generate(
                tokenized_prompt,
                shard_prng_key(subkey),
                self.params,
                gen_top_k,
                gen_top_p,
                temperature,
                cond_scale,
            )
            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]
            # decode images
            decoded_images = p_decode(encoded_images, self.vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for decoded_img in decoded_images:
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)

        return images
