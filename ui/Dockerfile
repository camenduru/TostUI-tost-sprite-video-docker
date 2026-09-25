FROM camenduru/tost-sprite-video

ENV s3_access_key_id=tost \
    s3_secret_access_key=tosttost \
    s3_endpoint_url=http://localhost:9000 \
    s3_region_name=us-east-1 \
    s3_bucket_name=tost \
    s3_bucket_folder=output \
    MINIO_ROOT_USER=tost \
    MINIO_ROOT_PASSWORD=tosttost \
    TOST_AI_SERVICES_URL=https://services.tost.ai \
    NEXT_PUBLIC_TOST_AI_PURCHASE_URL=https://buymeacoffee.com/camenduru/extras \
    NEXT_PUBLIC_TOST_AI_GUIDE_URL=https://guide.tost.ai \
    NEXT_PUBLIC_TOST_AI_FOLLOW_URL=https://x.com/tost_ai \
    NEXT_PUBLIC_TOST_AI_CONTAINS_URL=tost.ai \
    TOST_AI_S3_BASE_URL=https://s3.tost.ai/tost \
    NEXT_PUBLIC_TOST_UI_LOCAL_API_SERVICES=tost-sprite-video

USER root

RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g pnpm@10 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/minio/minio/releases/download/RELEASE.2025-04-22T22-12-26Z/minio.linux-amd64.RELEASE.2025-04-22T22-12-26Z -O /usr/local/bin/minio && \
    chmod +x /usr/local/bin/minio && \
    wget https://github.com/minio/mc/releases/download/RELEASE.2025-08-13T08-35-41Z/mc.linux-amd64.RELEASE.2025-08-13T08-35-41Z -O /usr/local/bin/mc && \
    chmod +x /usr/local/bin/mc

USER camenduru

RUN mkdir -p /content/data

WORKDIR /app
RUN git clone https://github.com/camenduru/TostUI .
RUN pnpm install && pnpm build

WORKDIR /content/ComfyUI

RUN echo '#!/bin/bash' > /content/start.sh && \
    echo '/usr/local/bin/minio server /content/data --console-address :9001 &' >> /content/start.sh && \
    echo 'sleep 10' >> /content/start.sh && \
    echo '/usr/local/bin/mc alias set local http://localhost:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"' >> /content/start.sh && \
    echo '/usr/local/bin/mc mb local/tost || true' >> /content/start.sh && \
    echo '/usr/local/bin/mc anonymous set public local/tost' >> /content/start.sh && \
    echo 'cd /app && pnpm start &' >> /content/start.sh && \
    echo 'python worker_runpod.py --rp_serve_api --rp_api_port 8000 --rp_api_host 0.0.0.0 --rp_log_level DEBUG --rp_debugger &' >> /content/start.sh && \
    echo 'wait -n' >> /content/start.sh

RUN chmod +x /content/start.sh
CMD ["/content/start.sh"]
