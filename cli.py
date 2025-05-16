import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download
from optimum.quanto import freeze, qint8, quantize
from PIL import Image
from torchvision.transforms.functional import normalize

from dreamo.dreamo_pipeline import DreamOPipeline
from dreamo.utils import (
    img2tensor,
    resize_numpy_image_area,
    resize_numpy_image_long,
    tensor2img,
)
from tools import BEN2


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference script for DreamO model")

    # Core I/O
    p.add_argument("--images", "-i", nargs="*", default=[], metavar="PATH",
                   help="Up to 4 reference images (PNG/JPG/...) in reading order")
    p.add_argument("--tasks", nargs="*", choices=["ip", "id", "style"],
                   help="Optional explicit task list for each reference image.")
    p.add_argument("--prompt", required=True, help="Text prompt")
    p.add_argument("--output", "-o", default="dreamo_output.png",
                   help="File to write the generated image (PNG)")

    # Generation params (default values match the Gradio demo)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=-1, help="-1 for random")

    # Advanced knobs (verbatim from demo, hidden in UI)
    p.add_argument("--ref_res", type=int, default=512)
    p.add_argument("--neg_prompt", default="")
    p.add_argument("--neg_guidance", type=float, default=3.5)
    p.add_argument("--true_cfg", type=float, default=1.0)
    p.add_argument("--cfg_start_step", type=int, default=0)
    p.add_argument("--cfg_end_step", type=int, default=0)
    p.add_argument("--first_step_guidance", type=float, default=0.0)

    # Runtime flags
    p.add_argument("--no-turbo", action="store_true", help="Disable FLUX‑turbo")
    p.add_argument("--int8", action="store_true", help="Quantise transformer/text encoder to int8")
    p.add_argument("--offload", action="store_true", help="Enable accelerate CPU offload mode")
    p.add_argument("--save-debug", action="store_true", help="Save preprocessing debug mosaic")
    return p


class DreamoGenerator:

    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bg_rm_model = BEN2.BEN_Base().to(self.device).eval()
        hf_hub_download(repo_id="PramaLLC/BEN2", filename="BEN2_Base.pth", local_dir="models")
        self.bg_rm_model.loadcheckpoints("models/BEN2_Base.pth")

        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            device=self.device,
        )

        model_root = "black-forest-labs/FLUX.1-dev"
        self.pipe = DreamOPipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16)
        self.pipe.load_dreamo_model(self.device, use_turbo=not args.no_turbo)

        if args.int8:
            print("[DreamO‑CLI] Quantising transformer + text encoder → int8…")
            for module in [self.pipe.transformer, self.pipe.text_encoder_2]:
                quantize(module, qint8)
                freeze(module)
            print("[DreamO‑CLI] Quantisation complete √")

        if args.offload:
            self.pipe.enable_model_cpu_offload()
            self.pipe.offload = True
        else:
            self.pipe = self.pipe.to(self.device)
            self.pipe.offload = False

        self.args = args

    def _ben_to(self, device):
        self.bg_rm_model.to(device)

    def _facex_to(self, device):
        self.face_helper.face_det.to(device)
        self.face_helper.face_parse.to(device)

    @torch.no_grad()
    def _align_face(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Return face‑aligned crop with bg masked, or None if no face."""
        self.face_helper.clean_all()
        self.face_helper.read_image(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            return None
        face = self.face_helper.cropped_faces[0]

        _in = img2tensor(face, bgr2rgb=True).unsqueeze(0) / 255.0
        _in = _in.to(self.device)
        parsing = self.face_helper.face_parse(normalize(_in, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing = parsing.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing == i for i in bg_label).bool()
        white = torch.ones_like(_in)
        feat = torch.where(bg, white, _in)
        return tensor2img(feat, rgb2bgr=False)

    def build_ref_cond(self, img: np.ndarray, task: str, idx: int, res: int) -> torch.Tensor:
        """Resize/crop/mask reference image and return conditioning tensor."""
        # Task‑specific preprocessing pipeline copied from the demo
        if task == "id":
            if self.args.offload:
                self._facex_to(self.device)
            img = resize_numpy_image_long(img, 1024)
            img = self._align_face(img)
            if img is None:
                raise RuntimeError("ID task requested but no face detected.")
            if self.args.offload:
                self._facex_to("cpu")
        elif task != "style":
            if self.args.offload:
                self._ben_to(self.device)
            img = self.bg_rm_model.inference(Image.fromarray(img))
            if self.args.offload:
                self._ben_to("cpu")
        if task != "id":
            img = resize_numpy_image_area(np.array(img), res * res)

        tensor = img2tensor(img, bgr2rgb=False).unsqueeze(0) / 255.0
        tensor = 2 * tensor - 1.0
        return {
            "img": tensor,
            "task": task,
            "idx": idx,
        }, img

    # ─────────────────────────────────────────────────────────────────────────
    def __call__(self, images: List[np.ndarray], tasks: List[str], params):
        ref_conds, debug_imgs = [], []
        for idx, (img, task) in enumerate(zip(images, tasks), start=1):
            cond, dbg = self.build_ref_cond(img, task, idx, params.ref_res)
            ref_conds.append(cond)
            debug_imgs.append(dbg)

        seed = params.seed if params.seed != -1 else torch.seed()
        gen = torch.Generator(device="cpu").manual_seed(seed)

        result = self.pipe(
            prompt=params.prompt,
            width=params.width,
            height=params.height,
            num_inference_steps=params.steps,
            guidance_scale=params.guidance,
            ref_conds=ref_conds,
            generator=gen,
            true_cfg_scale=params.true_cfg,
            true_cfg_start_step=params.cfg_start_step,
            true_cfg_end_step=params.cfg_end_step,
            negative_prompt=params.neg_prompt,
            neg_guidance_scale=params.neg_guidance,
            first_step_guidance_scale=(params.first_step_guidance or params.guidance),
        ).images[0]
        return result, debug_imgs, seed


def auto_detect_tasks(img_paths: List[Path], prompt: str, generator: DreamoGenerator) -> List[str]:
    """Very shallow heuristic: face → id, style prompt → style, else ip."""
    tasks = []
    is_style_prompt = prompt.strip().lower().startswith("generate a same style image")

    for p in img_paths:
        if is_style_prompt:
            tasks.append("style")
            continue
        # quick face check: use RetinaFace forward pass without alignment
        generator.face_helper.clean_all()
        generator.face_helper.read_image(cv2.imread(str(p)))
        generator.face_helper.get_face_landmarks_5(only_center_face=True)
        tasks.append("id" if len(generator.face_helper.landmarks) else "ip")
    return tasks


def main(argv: Optional[List[str]] = None):
    args = build_arg_parser().parse_args(argv)

    if len(args.images) == 0:
        sys.exit("[DreamO‑CLI] At least one --images path is required.")
    if len(args.images) > 4:
        sys.exit("[DreamO‑CLI] Maximum of four reference images supported.")

    img_paths = [Path(p) for p in args.images]
    for p in img_paths:
        if not p.exists():
            sys.exit(f"[DreamO‑CLI] Image not found: {p}")

    print("[DreamO‑CLI] Loading models")
    generator = DreamoGenerator(args)

    if args.tasks:
        if len(args.tasks) != len(img_paths):
            sys.exit("[DreamO‑CLI] --tasks count must match number of images, or omit entirely for auto‑detect.")
        tasks = args.tasks
    else:
        print("[DreamO‑CLI] Auto‑detecting tasks")
        tasks = auto_detect_tasks(img_paths, args.prompt, generator)

    images = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in img_paths]

    print("[DreamO‑CLI] Generating")
    out_img, debug_imgs, used_seed = generator(images, tasks, args)

    out_img.save(args.output)
    print(f"[DreamO‑CLI] Image saved to {args.output}  (seed={used_seed})")

    if args.save_debug:
        # tile debug images horizontally
        if debug_imgs:
            hmax = max(img.shape[0] for img in debug_imgs)
            canvas = np.concatenate([cv2.resize(i, (int(i.shape[1] * hmax / i.shape[0]), hmax)) for i in debug_imgs],
                                    axis=1)
            debug_path = Path(args.output).with_suffix(".debug.png")
            Image.fromarray(canvas).save(debug_path)
            print(f"[DreamO‑CLI] Debug mosaic saved to {debug_path}")


if __name__ == "__main__":
    main()
