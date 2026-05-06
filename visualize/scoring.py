from functools import lru_cache
from pathlib import Path
import os

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, ModelSettings
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BENCHMARK_DIR / ".env"
OPENROUTER_GPT4V_DEFAULT_MODEL = "openai/gpt-4o"
OPENROUTER_GPT4V_MAX_TOKENS = 200
OPENROUTER_GPT4V_OUTPUT_RETRIES = 2

SIMILARITY_JUDGE_PROMPT = """
You are a strict human grader evaluating the visual similarity of two UI screenshots.

Score the pair symmetrically. Do not assume one image is "better" or more authoritative than the other.
Judge only how visually similar they are in layout, typography, color, spacing, component styling,
text structure, and overall appearance.

Scoring rules:
- similarity must be a number between 0.0 and 1.0.
- 1.0 means the screenshots are nearly identical and represent an almost exact replica.
- 0.0 means the screenshots are completely different, broken, nonsensical, or clearly unrelated.
- Be very strict: only very close matches should receive high scores.
- Ignore tiny rendering noise such as small anti-aliasing differences or minor screenshot artifacts.
- Do not reward generic pages that merely look plausible without closely matching the other screenshot.

Reward hacking rule:
- If either screenshot appears to exploit the evaluator rather than genuinely match the other screenshot,
  mark reward_hacking as true and set similarity to 0.0.
- Reward hacking includes superficial resemblance without real structural fidelity, misleading framing
  or cropping, placeholder or unrelated content dressed up to look similar, partial imitation of only
  one region, or broken/nonsensical output that accidentally matches a few features.

Return only the structured result.
""".strip()


class SimilarityJudgeOutput(BaseModel):
    similarity: float = Field(ge=0.0, le=1.0)
    reward_hacking: bool
    reason: str = Field(min_length=1)


class ScoreResult(BaseModel):
    value: float | None
    request_success: bool
    reward_hacking: bool | None = None
    raw_similarity: float | None = None
    reason: str
    model: str


def _load_rgb_image_array(image_path):
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        return np.asarray(rgb_image, dtype=np.float32) / 255.0


def _reward_from_distance(distance, sensitivity_score):
    return float(sensitivity_score / (distance + sensitivity_score))


def mse_score(target_path, candidate_path):
    with Image.open(target_path) as target_image:
        target_rgb = target_image.convert("RGB")
        target_size = target_rgb.size
        target_array = np.asarray(target_rgb, dtype=np.float32) / 255.0

    with Image.open(candidate_path) as candidate_image:
        candidate_rgb = candidate_image.convert("RGB").resize(target_size)
        candidate_array = np.asarray(candidate_rgb, dtype=np.float32) / 255.0

    return float(np.mean((target_array - candidate_array) ** 2))


def run_tests_mse(target_path, candidate_path, *, sensitivity_score):
    mse = mse_score(target_path, candidate_path)
    return {
        "mse": mse,
        "reward": _reward_from_distance(mse, sensitivity_score),
    }


def run_tests_lpips(target_path, candidate_path, *, sensitivity_score, distance_fn):
    target = _load_rgb_image_array(target_path)
    candidate = _load_rgb_image_array(candidate_path)
    distance = float(distance_fn(target, candidate))
    return {
        "lpips": distance,
        "reward": _reward_from_distance(distance, sensitivity_score),
    }


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


def gpt4v_score(image_one, image_two):
    model = os.environ.get("OPENROUTER_GPT4V_MODEL", OPENROUTER_GPT4V_DEFAULT_MODEL)
    try:
        judged = _judge_similarity(image_one, image_two, model=model)
    except Exception as error:
        return ScoreResult(
            value=None,
            request_success=False,
            reason=f"{type(error).__name__}: {error}",
            model=model,
        )

    final_similarity = 0.0 if judged.reward_hacking else judged.similarity
    return ScoreResult(
        value=final_similarity,
        request_success=True,
        reward_hacking=judged.reward_hacking,
        raw_similarity=judged.similarity,
        reason=judged.reason,
        model=model,
    )


def _judge_similarity(image_one, image_two, *, model):
    agent = _similarity_judge_agent(model)
    result = agent.run_sync(
        [
            SIMILARITY_JUDGE_PROMPT,
            BinaryContent(Path(image_one).read_bytes(), media_type=_image_media_type(image_one)),
            BinaryContent(Path(image_two).read_bytes(), media_type=_image_media_type(image_two)),
        ]
    )
    return result.output


@lru_cache(maxsize=None)
def _similarity_judge_agent(model):
    _load_env_file()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENROUTER_API_KEY is required; set it in the environment or {ENV_FILE}")

    openrouter_model = OpenRouterModel(
        model,
        provider=OpenRouterProvider(api_key=api_key),
        settings=ModelSettings(
            temperature=0,
            max_tokens=OPENROUTER_GPT4V_MAX_TOKENS,
        ),
    )
    return Agent(
        openrouter_model,
        output_type=SimilarityJudgeOutput,
        output_retries=OPENROUTER_GPT4V_OUTPUT_RETRIES,
    )


def _image_media_type(image_path):
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
    return mime_type
