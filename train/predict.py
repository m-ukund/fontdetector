import argparse
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.ao.quantization as tq
import torch.ao.quantization.quantize_fx as qfx
from PIL import Image

# Filter out quantization warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*", 
                        category=UserWarning, module="torch._utils")


# ─────────────────────────── model defs ──────────────────────────────
class DeepFontAutoencoder(nn.Module):
    """Only the encoder part of DeepFont auto-encoder."""
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 12, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )


class DeepFont(nn.Module):
    """Classifier head on top of the frozen auto-encoder encoder."""
    def __init__(self, encoder: nn.Sequential, num_classes: int) -> None:
        super().__init__()
        self.ae_encoder = encoder
        for p in self.ae_encoder.parameters():
            p.requires_grad = False

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)

        hid = 4096
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 12 * 12, hid)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hid, hid)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hid, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        x = self.ae_encoder(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.flatten(x)
        x = self.drop1(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        return self.fc3(x)


# ───────────────────── checkpoint loaders ────────────────────────────
def _build_fp32(num_fonts: int) -> nn.Module:
    ae = DeepFontAutoencoder()
    return DeepFont(ae.encoder, num_fonts).eval()


def _load_fp32(ckpt: Path, num_fonts: int) -> nn.Module:
    mdl = _build_fp32(num_fonts)
    mdl.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return mdl.eval()


def _load_bf16(ckpt: Path, num_fonts: int) -> nn.Module:
    mdl = _build_fp32(num_fonts)
    mdl.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return mdl.to(torch.bfloat16).eval()


def _load_int8(ckpt: Path, num_fonts: int) -> nn.Module:
    torch.backends.quantized.engine = "fbgemm"
    base = _build_fp32(num_fonts)  # Model is created in .eval() mode

    example = torch.randn(1, 1, 105, 105)

    # Define a QConfig that matches fbgemm_qat_qconfig but avoids deprecated `reduce_range` in observer
    # For activations (e.g., after ReLU):
    # Use MovingAverageMinMaxObserver, target quint8, range [0, 127] (effectively 7-bit for fbgemm)
    # reduce_range=False as per warning's guidance.
    activation_qconfig = tq.FusedMovingAvgObsFakeQuantize.with_args(
        observer=tq.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=127, # Simulates reduce_range=True for an 8-bit type like quint8
        dtype=torch.quint8,
        reduce_range=False
    )
    # For weights:
    # Use MovingAveragePerChannelMinMaxObserver, target qint8, range [-128, 127]
    # qscheme is per_channel_symmetric for fbgemm. reduce_range is False by default.
    weight_qconfig = tq.FusedMovingAvgObsFakeQuantize.with_args(
        observer=tq.MovingAveragePerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=False # Explicitly set, though it's the default for weights
    )
    cfg = tq.QConfig(activation=activation_qconfig, weight=weight_qconfig)

    # Use QConfigMapping instead of a dictionary
    qconfig_mapping = tq.QConfigMapping().set_global(cfg)

    # prepare_qat_fx expects the model in train mode to insert observers/fake_quant modules
    base.train()
    # example_inputs is used for tracing the model
    prepared_model = qfx.prepare_qat_fx(base, qconfig_mapping, example_inputs=(example,))

    # Set model to eval mode for dummy pass and conversion
    prepared_model.eval()
    # Run a dummy forward pass. This helps initialize/run observers before conversion,
    # potentially silencing the "must run observer before calling calculate_qparams" warning.
    # The actual quantization parameters (scale, zero_point) will be loaded from the checkpoint.
    with torch.no_grad():
        prepared_model(example)

    # Convert the QAT-prepared model to a truly quantized model.
    # The model should be in eval mode for conversion.
    quant_mdl = qfx.convert_fx(prepared_model) # prepared_model is already in .eval()

    quant_mdl.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return quant_mdl.eval()


def load_model(quant: str, num_fonts: int) -> nn.Module:
    if quant == "none":
        return _load_fp32(Path(f"none_finetuned_n{num_fonts}.pt"), num_fonts)
    if quant == "amp_bf16":
        return _load_bf16(Path(f"amp_bf16_finetuned_n{num_fonts}.pt"),
                          num_fonts)
    if quant == "qat_int8":
        return _load_int8(Path(f"qat_int8_finetuned_n{num_fonts}.pt"),
                          num_fonts)
    if quant == "qat_int8_pc":
        return _load_int8(Path(f"qat_int8_pc_finetuned_n{num_fonts}.pt"),
                          num_fonts)
    raise ValueError(f"unsupported quant='{quant}'")


# ───────────────────────── image transforms ──────────────────────────
class ResizeHeight:
    def __init__(self, height: int = 105) -> None:
        self.height = height

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if h == 0:
            return Image.new(img.mode, (1, self.height), 255)
        new_w = max(1, round(self.height * w / h))
        return img.resize((new_w, self.height), Image.LANCZOS)


class Squeezing:
    def __init__(self, ratio: float = 2.5) -> None:
        self.ratio = ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        _, h = img.size
        return img.resize((max(1, round(h * self.ratio)), h), Image.LANCZOS)


class CenterPatch:
    def __init__(self, width: int = 105) -> None:
        self.width = width

    @staticmethod
    def _pad(img: Image.Image, width: int) -> Image.Image:
        w, h = img.size
        if w >= width:
            return img
        pad_colour = (255,) * (4 if img.mode == "RGBA"
                               else 3 if img.mode == "RGB" else 1)
        canvas = Image.new(img.mode, (width, h), pad_colour)
        canvas.paste(img, (0, 0))
        return canvas

    def __call__(self, img: Image.Image) -> Image.Image:
        img = self._pad(img, self.width)
        w, _ = img.size
        if w == self.width:
            return img
        sx = (w - self.width) // 2
        return img.crop((sx, 0, sx + self.width, img.height))


# ─────────────────────────── utilities ───────────────────────────────
def read_lines(font_path: str, num_fonts: int) -> list[str]:
    path = Path(font_path)
    if not path.exists():
        raise FileNotFoundError(f"font list file not found: {font_path}")
    with path.open("r") as f:
        return [ln for ln in f if "-" not in ln][:num_fonts]


# ───────────────────────────── main ──────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="path/to/image")
    ap.add_argument("--num_fonts", type=int, choices=(10, 100),
                    required=True, help="number of fonts: 10 or 100")
    ap.add_argument("--quant", default="none",
                    choices=("none", "amp_bf16", "qat_int8", "qat_int8_pc"),
                    help="checkpoint variant to load")
    args = ap.parse_args()

    # All inferences will be on the CPU.
    model = load_model(args.quant, args.num_fonts).to("cpu").eval()

    font_names = read_lines("fontlist.txt", args.num_fonts)

    img_path = Path(args.img)
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    img = Image.open(img_path).convert("RGB")

    test_pipe = T.Compose([
        ResizeHeight(105),
        Squeezing(),
        CenterPatch(105),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
    ])

    # Set dtype of input correctly
    x = test_pipe(img).unsqueeze(0).to("cpu")
    x = x.to(dtype=next(model.parameters()).dtype)

    with torch.inference_mode():
        idx = model(x).argmax(1).item()

    pred = (font_names[idx] if idx < len(font_names)
            else f"index {idx} out of {len(font_names)}")
    print("\nPredicted font:", pred)


if __name__ == "__main__":
    main()