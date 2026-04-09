#!/usr/bin/env python3
import argparse
import os
import inspect
from io import BytesIO
from urllib.request import urlopen

import torch
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from PIL import Image


def _patch_int8params_init() -> None:
    """
    Compatibility patch for some transformers/accelerate + bitsandbytes combos
    where Int8Params receives unexpected kwargs like `_is_hf_initialized`.
    """
    try:
        from bitsandbytes.nn import Int8Params
    except Exception:
        return

    sig = inspect.signature(Int8Params.__new__)
    if "_is_hf_initialized" not in sig.parameters:
        original_new = Int8Params.__new__

        def patched_new(
            cls,
            data=None,
            requires_grad=True,
            has_fp16_weights=False,
            CB=None,
            SCB=None,
            **kwargs,
        ):
            kwargs.pop("_is_hf_initialized", None)
            return original_new(
                cls,
                data=data,
                requires_grad=requires_grad,
                has_fp16_weights=has_fp16_weights,
                CB=CB,
                SCB=SCB,
            )

        Int8Params.__new__ = staticmethod(patched_new)


def _fix_special_tokens_cfg(tokenizer_source: str) -> dict:
    """Work around malformed `extra_special_tokens` in some model repos."""
    cfg = {"extra_special_tokens": {}}
    special_map_path = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub",
        f"models--{tokenizer_source.replace('/', '--')}",
        "snapshots",
    )
    if not os.path.isdir(special_map_path):
        return cfg

    try:
        snapshots = sorted(os.listdir(special_map_path), reverse=True)
        if not snapshots:
            return cfg
        latest = os.path.join(special_map_path, snapshots[0], "special_tokens_map.json")
        if not os.path.isfile(latest):
            return cfg
        import json

        with open(latest, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data.get("extra_special_tokens"), list):
            data["extra_special_tokens"] = {}
            with open(latest, "w", encoding="utf-8") as f:
                json.dump(data, f)
        return cfg
    except Exception:
        return cfg


def _load_image(image_ref: str) -> Image.Image:
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        with urlopen(image_ref) as r:
            raw = r.read()
        return Image.open(BytesIO(raw)).convert("RGB")
    return Image.open(image_ref).convert("RGB")


def build_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemma 4 26B A4B with 8-bit or 4-bit quantization.")
    parser.add_argument(
        "--model-id",
        default="coder3101/gemma-4-26B-A4B-it-heretic",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--tokenizer-id",
        default="google/gemma-4-26B-A4B-it",
        help="Tokenizer/processor model id fallback.",
    )
    parser.add_argument(
        "--prompt",
        default="Give me five practical ways to reduce local-LLM latency on consumer GPUs.",
        help="User prompt",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Image path or URL for multimodal prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument(
        "--quant",
        choices=["8bit", "4bit", "none"],
        default="8bit",
        help="Quantization mode.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["fp16", "bf16"],
        default="bf16",
        help="Compute dtype used for 4-bit mode.",
    )
    parser.add_argument(
        "--no-int8-cpu-offload",
        action="store_true",
        help="Disable fp32 CPU offload path for 8-bit mode.",
    )
    parser.add_argument(
        "--gpu-memory",
        default="22GiB",
        help="Per-GPU memory budget for accelerate device map, ex: 20GiB",
    )
    parser.add_argument(
        "--cpu-memory",
        default="64GiB",
        help="CPU RAM budget for offload, ex: 64GiB",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Gemma thinking mode in the chat template.",
    )
    return parser.parse_args()


def main() -> None:
    args = build_parser()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.quant == "8bit":
        _patch_int8params_init()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected. This script expects an NVIDIA GPU.")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_id}")

    processor = None
    tokenizer = None
    tokenizer_source = args.model_id
    try:
        processor = AutoProcessor.from_pretrained(args.model_id)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_id,
                **_fix_special_tokens_cfg(args.model_id),
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_id,
                **_fix_special_tokens_cfg(args.tokenizer_id),
            )
            tokenizer_source = args.tokenizer_id

    compute_dtype = torch.bfloat16 if args.compute_dtype == "bf16" else torch.float16

    if args.quant == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=not args.no_int8_cpu_offload,
        )
        model_dtype = torch.float16
    elif args.quant == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_dtype = compute_dtype
    else:
        quantization_config = None
        model_dtype = compute_dtype

    model_cls = AutoModelForMultimodalLM if args.image else AutoModelForCausalLM

    if args.quant == "8bit":
        max_memory = {
            0: args.gpu_memory,
            "cpu": args.cpu_memory,
        }
        model = model_cls.from_pretrained(
            args.model_id,
            dtype=model_dtype,
            device_map="auto",
            quantization_config=quantization_config,
            max_memory=max_memory,
        )
    elif args.quant == "4bit":
        model = model_cls.from_pretrained(
            args.model_id,
            dtype=model_dtype,
            device_map={"": 0},
            quantization_config=quantization_config,
        )
    else:
        model = model_cls.from_pretrained(
            args.model_id,
            dtype=model_dtype,
            device_map="auto",
        )

    if args.image:
        image = _load_image(args.image)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a concise, practical assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": args.prompt},
                ],
            },
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a concise, practical assistant."},
            {"role": "user", "content": args.prompt},
        ]

    if processor is not None:
        if args.image:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=inputs, images=image, return_tensors="pt").to(model.device)
        else:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            inputs = processor(text=text, return_tensors="pt").to(model.device)
        decode = processor.decode
    else:
        if args.image:
            raise SystemExit("Image mode requires a compatible AutoProcessor for this model.")
        print(f"Tokenizer source: {tokenizer_source}")
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
        decode = tokenizer.decode

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=True,
        )

    generated = decode(outputs[0][input_len:], skip_special_tokens=True)

    print("\n=== OUTPUT ===\n")
    print(generated)


if __name__ == "__main__":
    main()
