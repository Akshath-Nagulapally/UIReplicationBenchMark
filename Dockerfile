FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    BUN_INSTALL=/root/.bun \
    PATH="/root/.bun/bin:/root/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        chromium \
        curl \
        git \
        lsof \
        psmisc \
        procps \
        unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://bun.sh/install | bash \
    && npm install -g opencode-ai@latest \
    && opencode --version \
    && node --version \
    && npm --version \
    && bun --version \
    && uv --version \
    && chromium --version

WORKDIR /workspace

RUN bun create vite my-app --template react-ts --no-interactive \
    && cd /workspace/my-app \
    && bun install

SHELL ["/bin/bash", "-c"]

COPY harnesses/opencode/opencode.json /workspace/my-app/opencode.json
COPY harnesses /workspace/harnesses
COPY image_collection_utilities /workspace/image_collection_utilities
COPY image_sources.py /workspace/image_sources.py

RUN uv pip install --system playwright \
    && python -m playwright install chromium


WORKDIR /workspace/my-app

CMD ["bash", "-c", "bun run dev --host 0.0.0.0 >/tmp/bun-dev.log 2>&1 & bun_pid=$!; opencode serve >/tmp/opencode.log 2>&1 & opencode_pid=$!; trap 'kill \"$bun_pid\" \"$opencode_pid\" 2>/dev/null || true' EXIT; until curl -fsS http://127.0.0.1:5173 >/dev/null 2>&1; do kill -0 \"$bun_pid\" || { cat /tmp/bun-dev.log; exit 1; }; sleep 0.5; done; until curl -fsS http://127.0.0.1:4096/global/health >/dev/null 2>&1; do kill -0 \"$opencode_pid\" || { cat /tmp/opencode.log; exit 1; }; sleep 0.5; done; mkdir -p /workspace/output; cd /workspace && UI_REPLICATE_OUTPUT_DIR=/workspace/output python -m harnesses.opencode.run_opencode_task \"${UI_REPLICATE_PROMPT:-Recreate this UI}\" --image-url \"${UI_REPLICATE_TARGET_IMAGE_URL}\""]
