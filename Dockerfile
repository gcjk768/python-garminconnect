# Garmin Health Monitor: Python app + Claude Code CLI (claude -p) for the coaching.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*

# uid 1000 = "James Koh" on the NAS, so ./data stays owned by you, not root
RUN useradd -m -u 1000 app
USER app
ENV PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    GHM_CONFIG=/config/config.yaml \
    DISABLE_AUTOUPDATER=1

# native Claude Code CLI (no Node needed); auth comes from CLAUDE_CODE_OAUTH_TOKEN at runtime
RUN curl -fsSL https://claude.ai/install.sh | bash && claude --version

WORKDIR /app
COPY --chown=app pyproject.toml ./
COPY --chown=app garmin_health_monitor ./garmin_health_monitor
RUN pip install --user --no-cache-dir .

CMD ["python", "-m", "garmin_health_monitor", "run"]
