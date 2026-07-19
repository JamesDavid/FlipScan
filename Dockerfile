FROM python:3.11-slim

# ffmpeg for frame extraction; pango/gdk-pixbuf for weasyprint (reflowed PDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY flipscan ./flipscan
RUN pip install --no-cache-dir ".[ui,pdf]"

ENV FLIPSCAN_ROOT=/data
VOLUME /data
EXPOSE 8321

# same image serves CLI and GUI:
#   docker compose up                                  -> GUI on :8321
#   docker compose run --rm flipscan run /data/mybook  -> CLI
ENTRYPOINT ["flipscan"]
CMD ["ui", "--host", "0.0.0.0", "--port", "8321"]
