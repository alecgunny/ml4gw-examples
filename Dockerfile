FROM ghcr.io/ml4gw/pinto:main
EXPOSE 8888
WORKDIR /opt/ml4gw/examples

COPY environment.yaml /opt/ml4gw/examples
RUN source $CONDA_INIT && conda env create -f environment.yaml

COPY pyproject.toml poetry.lock poetry.toml utils/ /opt/ml4gw/examples
RUN source $CONDA_INIT \
        \
        && conda activate ml4gw-examples \
        \
        && /opt/conda/bin/poetry install

# do some gross and irresponsible permissions setting on
# root level directories for development purposes
RUN chmod 777 /opt/conda/envs/ml4gw-examples \
        \
        && mkidr /.local \
        \
        && chmod 777 /.local

COPY . /opt/ml4gw/examples
ENTRYPOINT ["pinto", "run", "jupyter", "notebook"]
CMD ["--NotebookApp.token=''", "--no-browser"]
