import os
from PIL import Image, ImageDraw, ImageFont

font_dir = "/data/google"
output_dir = "/data/synthetic"
text_samples = ["Sample", "Font123", "HelloWorld"]
img_size = (224, 224)

os.makedirs(output_dir, exist_ok=True)

for font_file in os.listdir(font_dir):
    if font_file.endswith(".ttf"):
        font_name = os.path.splitext(font_file)[0]
        font_path = os.path.join(font_dir, font_file)
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
