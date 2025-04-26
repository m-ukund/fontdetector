import os
from PIL import Image, ImageDraw, ImageFont

# Correct paths
font_dir = "/data/GoogleFonts"
output_dir = "/data/Synthetic"
text_samples = ["Sample", "Font123", "HelloWorld"]
img_size = (224, 224)

os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk(font_dir):
    for font_file in files:
        if font_file.endswith(".ttf"):
            font_name = os.path.splitext(font_file)[0]
            font_path = os.path.join(root, font_file)
            save_dir = os.path.join(output_dir, font_name)
            os.makedirs(save_dir, exist_ok=True)

            try:
                font = ImageFont.truetype(font_path, size=72)
                for idx, text in enumerate(text_samples):
                    img = Image.new("RGB", img_size, color="white")
                    draw = ImageDraw.Draw(img)
                    w, h = draw.textsize(text, font=font)
                    draw.text(((img_size[0] - w) / 2, (img_size[1] - h) / 2), text, fill="black", font=font)
                    img.save(os.path.join(save_dir, f"{idx}.png"))
            except Exception as e:
                print(f"Skipping {font_file}: {e}")

