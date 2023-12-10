import os
import random
import argparse 
from PIL import Image, ImageDraw

def main(args):
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    for i in range(args.n):
        # Create a black image
        img = Image.new("RGB", (args.wi, args.he), "black")
        draw = ImageDraw.Draw(img)

        for _ in range(args.N):
            # Randomly generate rectangle parameters
            rect_x = random.randint(0, args.wi)
            rect_y = random.randint(0, args.he)
            rnd_size = random.uniform(-0.1, 0.1)
            rect_width  = int((args.mh + rnd_size) * args.wi)
            rect_height = int((args.mw + rnd_size) * args.he)
            # Ensure the rectangle stays inside the image
            rect_x = min(rect_x, args.wi - rect_width)
            rect_y = min(rect_y, args.he - rect_height)

            print(rect_width, rect_height)
            # Draw the rectangle on the image
            draw.rectangle([rect_x, rect_y, rect_x + rect_width, rect_y + rect_height], fill="white")

        # Save the image to the output directory
        img.save(os.path.join(args.o, f"mask_{i}.png"))


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",  type=int,  required=True, help="The number of image to generate")
    parser.add_argument("-o",  type=str,  required=True, help="The output directory")
    parser.add_argument("-he",  type=int,  required=False, default=512)
    parser.add_argument("-wi",  type=int,  required=False, default=512)
    parser.add_argument("-N",  type=int,  required=False, default=1, help="The number of rectangles per image")
    parser.add_argument("-mh", type=float, required=False, default=0.3, help="Mean h of the rectangles")
    parser.add_argument("-mw", type=float, required=False, default=0.2, help="Mean w of the rectangles")
    args = parser.parse_args()
    main(args)

if __name__=='__main__':
    cli()