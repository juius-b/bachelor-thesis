from PIL import Image

with Image.open("chest.png") as im:
    width, height = 256, 256
    im_resized = im.resize((width, height), Image.Resampling.BICUBIC)

    im_resized.save("chest-resized.png")

