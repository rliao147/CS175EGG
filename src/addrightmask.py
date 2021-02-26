from PIL import Image
import random

size = 40

numImages = 5101
for num in range(1, numImages):
    x = random.randint(0, 255 - size)
    y = random.randint(0, 255 - size)
    im = Image.open("./img/" + str(num) + ".jpg")
    pixels = im.load()
    for i in range(size):
       for j in range(size):
           pixels[i + x, j + y] = (0, 0, 0)
    im.save("./imgmasked/" + str(num) + "masked.jpg")