FROM ghcr.io/astral-sh/uv:python3.13-bookworm

# Install Docker CLI (purple agent launches sibling containers for SWE-bench, CyberGym, etc.)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg poppler-utils && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
    > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y --no-install-recommends docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password agent
RUN groupadd -f docker && usermod -aG docker agent
RUN chmod 777 /var/run

USER agent
WORKDIR /home/agent

COPY --chown=agent pyproject.toml uv.lock ./
COPY --chown=agent src src

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

# Pre-build the OfficeQA BM25 index so it loads in seconds at runtime
# instead of taking 2-5 minutes to download, chunk, and index 697 files.
RUN cd src && uv run python -m skills.officeqa_build_index

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 9009
