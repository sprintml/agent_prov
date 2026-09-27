"""Load and manage HuggingFace models for tool-calling experiments."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Pre-import gptqmodel so its module-level constants are initialised in a
# normal CPU context rather than inside accelerate's init_empty_weights()
# context manager (which makes torch.tensor() land on meta device, causing
# a RuntimeError when gptqmodel tries to call .item() on them).
# Transformers 5.5+ uses gptqmodel as the backend for both AWQ and GPTQ.
try:
    import gptqmodel  # noqa: F401
    # optimum 2.1.0 references BACKEND.EXLLAMA_V1 which gptqmodel 6.0+ renamed
    # to EXLLAMA_V2. Add the alias so optimum's post_init_model doesn't crash.
    from gptqmodel.models.loader import BACKEND
    if not hasattr(BACKEND, "EXLLAMA_V1") and hasattr(BACKEND, "EXLLAMA_V2"):
        BACKEND.EXLLAMA_V1 = BACKEND.EXLLAMA_V2
except Exception:
    pass


def detect_device():
    """Auto-detect best available device.

    Returns "auto" for CUDA so accelerate distributes across all visible GPUs.
    This handles both single-GPU (CUDA_VISIBLE_DEVICES=0) and multi-GPU setups.
    """
    if torch.cuda.is_available():
        return "auto"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def load_model(model_id: str, device: str = "auto", dtype_override: str | None = None,
               load_in_8bit: bool = False, load_in_4bit: bool = False):
    """Load a model and tokenizer from HuggingFace.

    Args:
        model_id: HuggingFace model identifier.
        device: Device to load on. "auto" will detect best available.
        dtype_override: Force a specific dtype ("float16", "bfloat16", "float32").
            When None, auto-selects based on device.
        load_in_8bit: Use bitsandbytes 8-bit quantization (w8a16).
        load_in_4bit: Use bitsandbytes NF4 4-bit quantization.

    Returns:
        Tuple of (model, tokenizer).
    """
    if device == "auto":
        device = detect_device()

    print(f"Loading {model_id} on {device}...")

    # Phi-4 ships custom code incompatible with transformers 5.x,
    # but Phi-4 is natively supported - skip remote code for it.
    # GLM-4 and InternLM need trust_remote_code=True for their custom tokenizers.
    no_remote = ["phi"]
    needs_remote = ["glm", "internlm", "thudm"]
    model_lower = model_id.lower()
    if any(x in model_lower for x in needs_remote):
        trust_remote = True
    elif any(x in model_lower for x in no_remote):
        trust_remote = False
    else:
        trust_remote = True

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote)

    if dtype_override:
        dtype = _DTYPE_MAP.get(dtype_override)
        if dtype is None:
            raise ValueError(f"Unknown dtype_override '{dtype_override}'. "
                             f"Valid: {sorted(_DTYPE_MAP.keys())}")
    elif device == "cpu":
        dtype = torch.float32
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    kwargs = dict(
        device_map=device,
        trust_remote_code=trust_remote,
    )

    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
    else:
        kwargs["dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except AttributeError as exc:
        if "all_tied_weights_keys" in str(exc):
            # Transformers 5.5.x changed the tied-weights property API; custom-code
            # models (GLM-4, some InternLM versions) that override __getattr__ raise
            # AttributeError for the new property. low_cpu_mem_usage=False loads the
            # model directly without meta tensors, bypassing both infer_auto_device_map
            # and _move_missing_keys_from_meta_to_device where the error occurs.
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["device_map"] = None
            fallback_kwargs["low_cpu_mem_usage"] = False
            model = AutoModelForCausalLM.from_pretrained(model_id, **fallback_kwargs)
            if torch.cuda.is_available():
                model = model.to("cuda")
        else:
            raise
    model.eval()

    print(f"Loaded {model_id} ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)")
    return model, tokenizer
