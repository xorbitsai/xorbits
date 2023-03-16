ARG PYTHON_VERSION=3.9
ARG BASE_CONTAINER=xprobe/xorbits:base-py${PYTHON_VERSION}
FROM ${BASE_CONTAINER}

# Run this under `xorbits` dir
COPY . /opt/xorbits/

SHELL ["/bin/bash", "-c"]
ARG PYTHON_VERSION=3.9
RUN cd /opt/xorbits/python && \
    if [ "$PYTHON_VERSION" == "3.11" ] ; \
    then CFLAGS="-DCYTHON_FAST_THREAD_STATE=0" python setup.py build_ext -i ; \
    else python setup.py build_ext -i ; fi && \
    npm cache clean --force && \
    python setup.py build_web && \
    rm -rf /opt/xorbits/python/xorbits/web/ui/node_modules

RUN mkdir -p /srv

WORKDIR /srv

RUN cp /opt/xorbits/python/xorbits/deploy/oscar/file-logging.conf /srv/logging.conf && \
    cp /opt/xorbits/python/xorbits/deploy/docker/entrypoint.sh /srv/entrypoint.sh && \
    cp /opt/xorbits/python/xorbits/deploy/docker/install.sh /srv/install.sh && \
    chmod a+x /srv/*.sh

ENV PYTHONPATH "${PYTHONPATH}:/opt/xorbits:/opt/xorbits/python"

ENTRYPOINT [ "/srv/entrypoint.sh" ]
