import torch
from imagen_pytorch import Unet, Imagen
import torchvision
from PIL import Image
import os
import numpy as np


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    ndarr = np.uint8(255*ndarr)
    im = Image.fromarray(ndarr)
    im.save(path)


def main():
    # unet for imagen

    unet1 = Unet(
        dim=32,
        cond_dim=512,
        dim_mults=(1, 2, 4, 8),
        num_resnet_blocks=3,
        layer_attns=(False, True, True, True),
        layer_cross_attns=(False, True, True, True)
    )

    unet2 = Unet(
        dim=32,
        cond_dim=512,
        dim_mults=(1, 2, 4, 8),
        num_resnet_blocks=(2, 4, 8, 8),
        layer_attns=(False, False, False, True),
        layer_cross_attns=(False, False, False, True)
    )

    # imagen, which contains the unets above (base unet and super resoluting ones)

    imagen = Imagen(
        unets=(unet1, unet2),
        image_sizes=(64, 256),
        timesteps=1000,
        cond_drop_prob=0.1
    ).cuda()

    # mock images (get a lot of this) and text encodings from large T5
    # Change this with your actual dataset
    text_embeds = torch.randn(4, 256, 768).cuda()
    images = torch.randn(4, 3, 256, 256).cuda()

    # feed images into imagen, training each unet in the cascade
    epochs = 5
    for epoch in epochs:
        for i in (1, 2):
            loss = imagen(images, text_embeds=text_embeds, unet_number=i)
            loss.backward()

    # do the above for many many many many steps
    # now you can sample an image based on the text embeddings from the cascading ddpm

    images = imagen.sample(texts=[
        'a whale breaching from afar',
        'young girl blowing out candles on her birthday cake',
        'fireworks with blue and green sparkles'
    ], cond_scale=3.)

    print(images.shape)  # (3, 3, 256, 256)
    save_images(images, os.path.join('./results.jpg'))


if __name__ == '__main__':
    main()
