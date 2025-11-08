from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.utils.config import ViralIdea, get_settings
from src.utils.logger import get_logger

try:
    from diffusers import StableDiffusionPipeline
    import torch
except ImportError:  # pragma: no cover - optional dependency
    StableDiffusionPipeline = None  # type: ignore
    torch = None  # type: ignore


@dataclass
class ImageResult:
    image_path: Path
    prompt: str
    seed: int


class TextToImageGenerator:
    """Wraps an open-source text-to-image pipeline with optional fallback."""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1",
        device: Optional[str] = None,
    ) -> None:
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)
        self.model_id = model_id
        self.device = device or self._auto_device()
        self.pipe = self._load_pipeline()

    def _load_pipeline(self):
        if StableDiffusionPipeline is None:
            self.logger.warning("diffusers not installed; image generation disabled.")
            return None

        self.logger.info("Loading diffusion pipeline %s on %s", self.model_id, self.device)
        torch_dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32  # type: ignore[attr-defined]
        pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch_dtype)
        pipe = pipe.to(self.device)
        if self.device == "mps":
            pipe.enable_attention_slicing()
        pipe.safety_checker = None
        return pipe

    def generate_image(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        if not self.pipe:
            raise RuntimeError("Diffusion pipeline not available. Install diffusers and model weights.")

        seed = seed or random.randint(0, 1_000_000)
        generator = torch.Generator(device=self.device).manual_seed(seed) if torch else None
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
            height=self.settings.image_output_size,
            width=self.settings.image_output_size,
        )
        image = result.images[0]
        return image

    def _auto_device(self) -> str:
        if not torch:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"  # type: ignore[attr-defined]
        return "cpu"


class ImageComposer:
    """Adds text overlays to generated images."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)
        self.font = self._load_font()

    def _load_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        for path in font_paths:
            if Path(path).exists():
                return ImageFont.truetype(path, size=int(self.settings.image_output_size * 0.06))
        self.logger.warning("Custom font not found; using default PIL font.")
        return ImageFont.load_default()

    def compose(self, image: Image.Image, idea: ViralIdea, output_path: Path) -> Path:
        img = image.convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Dark gradient at bottom
        gradient_height = int(img.height * 0.45)
        gradient = Image.new("L", (1, gradient_height), color=0xFF)
        for y in range(gradient_height):
            gradient.putpixel((0, y), int(255 * (y / gradient_height)))
        alpha_gradient = gradient.resize((img.width, gradient_height))
        black = Image.new("RGBA", (img.width, gradient_height), color=(0, 0, 0, 255))
        black.putalpha(alpha_gradient)
        overlay.paste(black, (0, img.height - gradient_height))

        combined = Image.alpha_composite(img, overlay)

        # Headline text
        text_area_margin = int(img.width * 0.06)
        text_width = img.width - 2 * text_area_margin
        headline = idea.headline.upper()
        wrapped = self._wrap_text(headline, draw, text_width)

        text_y = img.height - gradient_height + text_area_margin
        for line in wrapped:
            line_bbox = draw.textbbox((0, 0), line, font=self.font)
            line_height = line_bbox[3] - line_bbox[1]
            draw = ImageDraw.Draw(combined)
            # Shadow
            shadow_offset = int(self.settings.image_output_size * 0.002)
            draw.text(
                (text_area_margin + shadow_offset, text_y + shadow_offset),
                line,
                font=self.font,
                fill=(0, 0, 0, 255),
            )
            draw.text(
                (text_area_margin, text_y),
                line,
                font=self.font,
                fill=(255, 255, 255, 255),
            )
            text_y += line_height + int(line_height * 0.2)

        combined = combined.convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.save(output_path, format="JPEG", quality=95)
        return output_path

    def _wrap_text(self, text: str, draw: ImageDraw.ImageDraw, max_width: int):
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            test_line = f"{current} {word}".strip()
            width = draw.textlength(test_line, font=self.font)
            if width <= max_width:
                current = test_line
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines


class CreativeEngine:
    """Combines generation and composition into a single workflow."""

    def __init__(self, generator: TextToImageGenerator | None = None, composer: ImageComposer | None = None):
        self.settings = get_settings()
        self.generator = generator or TextToImageGenerator()
        self.composer = composer or ImageComposer()
        self.logger = get_logger(self.__class__.__name__)

    def create_asset(self, idea: ViralIdea, output_dir: Path | None = None) -> ImageResult:
        prompt = f"{idea.angle}. Highlight futuristic, cinematic lighting, editorial photography."
        seed = random.randint(0, 1_000_000)
        base_image = self.generator.generate_image(prompt=prompt, seed=seed)

        output_dir = output_dir or self.settings.output_dir
        filename = f"{idea.headline.lower().replace(' ', '_')[:40]}_{seed}.jpg"
        safe_filename = "".join(c for c in filename if c.isalnum() or c in ("_", "-", "."))
        output_path = output_dir / safe_filename
        image_path = self.composer.compose(base_image, idea, output_path)

        self.logger.info("Created image asset at %s", image_path)
        return ImageResult(image_path=image_path, prompt=prompt, seed=seed)


