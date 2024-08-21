FROM daskdev/dask-jobqueue:slurm

SHELL ["/bin/bash", "-c"]

# activate conda env
RUN echo "source /opt/anaconda/bin/activate dask-jobqueue" >> ~/.bashrc
ENV PATH /opt/anaconda/envs/dask-jobqueue/bin:$PATH
