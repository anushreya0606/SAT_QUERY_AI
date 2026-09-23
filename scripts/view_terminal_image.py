"""
Terminal Image Viewer for Remote Sensing Imagery
================================================
Renders any satellite or aerial image directly in the terminal using 
24-bit ANSI TrueColor half-blocks.

Usage:
    python scripts/view_terminal_image.py <path_to_image> [width]

Example:
    python scripts/view_terminal_image.py data/bhoonidhi/cartosat/images/scene_0000.png 60
"""

import os
import sys
from PIL import Image

# Ensure stdout handles UTF-8 block characters on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def display_image_in_terminal(image_path: str, width: int = 50):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error opening image: {e}")
        return

    orig_w, orig_h = img.size
    # Terminal characters are roughly twice as tall as they are wide
    aspect = orig_h / orig_w
    height = int(width * aspect * 0.5)
    
    # Resize to width x (height * 2) so 2 vertical pixels fit in one half-block character
    img_resized = img.resize((width, max(2, height * 2)), Image.Resampling.BILINEAR)

    print(f"\n🛰️  Displaying: {os.path.basename(image_path)} ({orig_w}x{orig_h} px)")
    print("=" * width)

    for y in range(0, height * 2, 2):
        row = []
        for x in range(width):
            r_top, g_top, b_top = img_resized.getpixel((x, y))
            if y + 1 < height * 2:
                r_bot, g_bot, b_bot = img_resized.getpixel((x, y + 1))
            else:
                r_bot, g_bot, b_bot = (0, 0, 0)
            
            # Upper block char: foreground = top pixel color, background = bottom pixel color
            row.append(f"\033[38;2;{r_top};{g_top};{b_top}m\033[48;2;{r_bot};{g_bot};{b_bot}m▀")
        row.append("\033[0m")
        print("".join(row))

    print("=" * width + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default test image if no argument provided
        default_img = "data/bhoonidhi/cartosat/images/scene_0000.png"
        if os.path.exists(default_img):
            display_image_in_terminal(default_img, width=50)
        else:
            print("Usage: python scripts/view_terminal_image.py <path_to_image> [width]")
    else:
        path = sys.argv[1]
        w = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        display_image_in_terminal(path, width=w)
