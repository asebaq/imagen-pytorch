import torch
from imagen_pytorch import Unet, Imagen, ImagenTrainer
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
        text_encoder_name='t5-large',
        image_sizes=(64, 256),
        timesteps=1000,
        cond_drop_prob=0.1
    ).cuda()

    # wrap imagen with the trainer class

    trainer = ImagenTrainer(imagen)

    # mock images (get a lot of this) and text encodings from large T5

    text_embeds = torch.randn(64, 256, 1024).cuda()
    images = torch.randn(64, 3, 256, 256).cuda()

    # feed images into imagen, training each unet in the cascade

    loss = trainer(
        images,
        text_embeds=text_embeds,
        unet_number=1,            # training on unet number 1 in this example, but you will have to also save checkpoints and then reload and continue training on unet number 2
        # auto divide the batch of 64 up into batch size of 4 and accumulate gradients, so it all fits in memory
        max_batch_size=4
    )

    trainer.update(unet_number=1)

    # do the above for many many many many steps
    # now you can sample an image based on the text embeddings from the cascading ddpm

    images = trainer.sample(texts=[
        'a puppy looking anxiously at a giant donut on the table',
        'the milky way galaxy in the style of monet'
    ], cond_scale=3.)

    print(images.shape)  # (3, 3, 256, 256)
    save_images(images, os.path.join('./results.jpg'))


if __name__ == '__main__':
    main()
