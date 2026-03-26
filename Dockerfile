FROM python:3.11-slim AS builder

WORKDIR /build

# C++ toolchain required to build hnswlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir hatchling

COPY pyproject.toml .
COPY recommender/ recommender/

# Install into a prefix we can copy cleanly
RUN pip install --no-cache-dir --prefix=/install .


FROM python:3.11-slim

# tini for proper PID 1 signal handling
RUN apt-get update && apt-get install -y --no-install-recommends tini && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local

# Non-root user; create /models with correct ownership before declaring the volume
RUN useradd -m -u 1000 recommender \
    && mkdir -p /models \
    && chown recommender:recommender /models

VOLUME ["/models"]

USER recommender

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--", "recommender"]
CMD ["api"]
