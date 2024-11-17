FROM continuumio/miniconda3:24.9.2-0 AS base

ARG PYTHON_VERSION=3.9
SHELL ["/bin/bash", "-c"]

RUN apt-get -y update \
    && apt-get install -y \
      curl \
      procps \
      gcc \
      g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda install python=${PYTHON_VERSION} \
    nodejs=20.17 \
    conda-forge::mkl \
    conda-forge::libnuma \
    && conda clean --all -f -y
# TODO: UCXX is not mature enough for production, add it back when it's ready

RUN pip install -U \
    numpy \
    scipy \
    pandas \
    numexpr \
    psutil \
    scikit-learn \
    sqlalchemy \
    tornado \
    xoscar \
    pyarrow \
    cloudpickle \
    azure-storage-blob \
    adlfs \
    fsspec \
    s3fs \
    pyopenssl \
    datasets \
    python-kubernetes \
    jax \
    uvloop \
    Cython

RUN if [ "$PYTHON_VERSION" == "3.9" ] ; then \
    pip install -U 'setuptools<64' ; fi

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

RUN cp /opt/xorbits/python/xorbits/deploy/docker/entrypoint.sh /srv/entrypoint.sh && \
    cp /opt/xorbits/python/xorbits/deploy/docker/install.sh /srv/install.sh && \
    chmod a+x /srv/*.sh

ENV PYTHONPATH "${PYTHONPATH}:/opt/xorbits:/opt/xorbits/python"

ENTRYPOINT [ "/srv/entrypoint.sh" ]
