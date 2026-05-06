from functools import lru_cache
from pathlib import Path
import base64
import json
import os
import urllib.error
import urllib.request

import numpy as np
from PIL import Image


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BENCHMARK_DIR / ".env"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_GPT4V_DEFAULT_MODEL = "openai/gpt-4o"


def _load_env_file(env_file=None):
    env_file = ENV_FILE if env_file is None else env_file
    if not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def _load_rgb_image_array(image_path, *, size=None):
    with Image.open(image_path) as image:
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.BICUBIC)
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _load_image_pair(image_one, image_two):
    with Image.open(image_one) as first_image:
        target_size = first_image.size
        first = np.asarray(first_image.convert("RGB"), dtype=np.float32) / 255.0
    second = _load_rgb_image_array(image_two, size=target_size)
    return first, second


def _reward_from_distance(distance, sensitivity_score):
    if sensitivity_score <= 0:
        raise ValueError("sensitivity_score must be positive")

    return 1.0 / (1.0 + distance / sensitivity_score)


def mse_score(image_one, image_two):
    first, second = _load_image_pair(image_one, image_two)
    return float(np.mean((first - second) ** 2))


def lpips_score(image_one, image_two):
    first, second = _load_image_pair(image_one, image_two)
    return float(_lpips_distance(first, second))


def gpt4v_score(image_one, image_two):
    verdict = _llm_as_judge_verdict(
        image_one,
        image_two,
        model=os.environ.get("OPENROUTER_GPT4V_MODEL", OPENROUTER_GPT4V_DEFAULT_MODEL),
    )
    return 0.0 if verdict == "PASS" else 1.0


def run_tests_mse(image_one, image_two, sensitivity_score):
    mse = mse_score(image_one, image_two)

    return {"mse": mse, "reward": _reward_from_distance(mse, sensitivity_score)}


def run_tests_lpips(image_one, image_two, sensitivity_score, distance_fn=None):
    if distance_fn:
        first, second = _load_image_pair(image_one, image_two)
        lpips_distance = float(distance_fn(first, second))
    else:
        lpips_distance = lpips_score(image_one, image_two)

    return {"lpips": lpips_distance, "reward": _reward_from_distance(lpips_distance, sensitivity_score)}


def _llm_as_judge_verdict(image_one, image_two, *, model):
    response_payload = _openrouter_chat_completion(
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are judging whether a generated UI screenshot successfully matches a target UI screenshot. "
                                "The first image is the target. The second image is the candidate. The dimensions for candidate are fixed and out of the candidate's control, so judge accordingly. "
                                "Reply with exactly one word: PASS if the candidate is a sufficiently faithful UI match, "
                                "otherwise FAIL."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_to_data_url(image_one)},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_to_data_url(image_two)},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 16,
        }
    )
    content = _response_message_text(response_payload)
    verdict = _extract_pass_fail(content)
    if verdict is None:
        raise RuntimeError(f"OpenRouter judge did not return PASS/FAIL: {content!r}")
    return verdict


def _openrouter_chat_completion(payload):
    _load_env_file()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENROUTER_API_KEY is required; set it in the environment or {ENV_FILE}")

    request = urllib.request.Request(
        OPENROUTER_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter request failed with status {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"OpenRouter request failed: {error.reason}") from error


def _image_to_data_url(image_path):
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        mime_type = Image.MIME.get(image.format)
    if mime_type is None:
        suffix = image_path.suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(suffix, "application/octet-stream")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _response_message_text(response_payload):
    try:
        message = response_payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Unexpected OpenRouter response shape: {response_payload!r}") from error

    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return " ".join(text_parts)
    return str(content)


def _extract_pass_fail(text):
    normalized = str(text).strip().upper()
    tokens = normalized.replace(".", " ").replace(",", " ").split()
    for token in tokens:
        if token in {"PASS", "FAIL"}:
            return token
    return None


def _lpips_distance(first, second):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("run_tests_lpips requires the optional 'torch' and 'lpips' packages") from error

    first_tensor = _lpips_tensor(first, torch)
    second_tensor = _lpips_tensor(second, torch)

    with torch.no_grad():
        return _lpips_model()(first_tensor, second_tensor).item()


def _lpips_tensor(image_array, torch_module):
    return torch_module.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


@lru_cache(maxsize=1)
def _lpips_model():
    try:
        import lpips
    except ImportError as error:
        raise RuntimeError("run_tests_lpips requires the optional 'lpips' package") from error

    return lpips.LPIPS(net="alex")
