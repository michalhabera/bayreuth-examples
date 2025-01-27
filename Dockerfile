FROM mhabera/fenicsx-examples:latest

ARG NB_USER=jupyter
ARG NB_UID=1000

RUN pip install imageio

RUN id -nu ${NB_UID} && userdel --force $(id -nu ${NB_UID}) || true; \
    useradd -m ${NB_USER} -u ${NB_UID}
ENV HOME=/home/${NB_USER}

WORKDIR ${HOME}
COPY --chown=${NB_UID} . ${HOME}

USER ${NB_USER}
ENTRYPOINT []
