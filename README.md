# FastSDXL

This is an efficient implementation of the UNet in Stable-Diffusion-XL. I reconstructed the architecture of UNet and found this implementation is faster than others ([Diffusers](https://github.com/huggingface/diffusers), [Fooocus](https://github.com/lllyasviel/Fooocus), etc.).

## Usage

The code is headless, you only need to install a few packages. I developed this project based on `diffusers==0.21.3`. If you find it cannot run with another version of `diffusers`, please open an issue and tell me.

```
pip install diffusers safetensors torch gradio
```

We provide a demo here:

```
python launch.py
```

## Efficnency

I tested my code using NVidia 3060 laptop (6G, 85W). The resolution is 1024*1024, and the model is converted to float16 format.

* Diffusers: CUDA OOM
* Fooocus: 1.78s/it
* FastSDXL (ours): 1.17s/it
