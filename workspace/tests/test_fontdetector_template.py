import os
import random
from PIL import Image
import itertools

TEMPLATE_DIR = "templates"

def compose_template(font_path, bg_path=None, extra_path=None):
    font_img = Image.open(font_path).convert("RGBA")
    bg = Image.new("RGBA", font_img.size, (255, 255, 255, 255)) if bg_path is None else Image.open(bg_path).convert("RGBA")
    bg_w, bg_h = bg.size
    y_offset = int(bg_h * 0.05)

    font_img = font_img.resize((int(bg_w * 0.5), int(bg_h * 0.5)))
    fd_w, fd_h = font_img.size

    if extra_path:
        extra = Image.open(extra_path).convert("RGBA")
        extra = extra.resize((int(bg_w * 0.35), int(bg_h * 0.35)))
        ex_w, ex_h = extra.size
        bg.paste(extra, (bg_w - ex_w, bg_h - ex_h - y_offset), extra)

    bg.paste(font_img, ((bg_w - fd_w) // 2, bg_h - fd_h - y_offset), font_img)
    return bg.convert("RGB")

# Require 80% accuracy on composed template variants
def test_template_permutations(model, predict, font_list):
    font_dir_root = os.path.join(TEMPLATE_DIR, "fonts")
    backgrounds = os.listdir(os.path.join(TEMPLATE_DIR, "background"))
    extras = os.listdir(os.path.join(TEMPLATE_DIR, "extras"))

    total_tests = 0
    passed_tests = 0
    failures = []

    for font_name in font_list:
        font_dir = os.path.join(font_dir_root, font_name)
        if not os.path.exists(font_dir):
            continue
        font_imgs = [f for f in os.listdir(font_dir) if f.endswith(".png")]
        if not font_imgs:
            continue

        selected_img = random.choice(font_imgs)
        font_path = os.path.join(font_dir, selected_img)
        class_index = font_list.index(font_name)

        combinations = itertools.product(backgrounds, extras)
        for bg, extra in combinations:
            total_tests += 1
            img = compose_template(
                font_path,
                bg_path=os.path.join(TEMPLATE_DIR, "background", bg),
                extra_path=os.path.join(TEMPLATE_DIR, "extras", extra)
            )
            pred = predict(model, img)
            if pred == class_index:
                passed_tests += 1
            else:
                failures.append((font_name, bg, extra, font_list[pred] if pred < len(font_list) else "invalid"))

    pass_ratio = passed_tests / total_tests if total_tests > 0 else 0
    assert pass_ratio >= 0.80, f"Only {passed_tests}/{total_tests} ({pass_ratio*100:.1f}%) template permutations passed. Failures: {failures}"
