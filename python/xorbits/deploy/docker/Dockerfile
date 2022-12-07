ARG BASE_CONTAINER=xprobe/xorbits:base
FROM ${BASE_CONTAINER}

# Run this under `xorbits` dir
COPY . /opt/xorbits/

RUN cd /opt/xorbits/python && \
    python setup.py build_ext -i && \
    npm cache clean --force && \
    python setup.py build_web && \
    rm -rf /opt/xorbits/python/xorbits/ui/node_modules

RUN mkdir -p /srv

WORKDIR /srv

RUN cp /opt/xorbits/third_party/_mars/mars/deploy/oscar/file-logging.conf /srv/logging.conf && \
    cp /opt/xorbits/python/xorbits/deploy/docker/entrypoint.sh /srv/entrypoint.sh && \
    chmod a+x /srv/*.sh

ENV PYTHONPATH "${PYTHONPATH}:/opt/xorbits:/opt/xorbits/python"

ENTRYPOINT [ "/srv/entrypoint.sh" ]
