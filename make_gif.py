import sys
import os
from PIL import Image

IMAGE_DIMENSION = 28

# Create a gif file of all the png files found in the provided folder.
def make_gif(folder):
    png_paths = [folder + file for file in os.listdir(folder) if file.split(".")[-1] == "png"]
    # Sort the files according to their integer file name.
    png_paths.sort(key=lambda path: int(path.split("/")[-1].split(".")[0]))
    frames = [Image.open(png) for png in png_paths]
    im = Image.new('L', (IMAGE_DIMENSION, IMAGE_DIMENSION * 10))
    im.save(folder + 'out.gif', save_all=True, append_images=frames, duration=150)

if __name__ == "__main__":
    make_gif(sys.argv[1])
