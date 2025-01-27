FROM dolfinx/dolfinx:v0.9.0

ENV LIBGL_ALWAYS_SOFTWARE=1
ENV DISPLAY=":99.0"
ENV PYVISTA_TRAME_SERVER_PROXY_PREFIX="/proxy/"

RUN pip install jupytext jupyter scipy

RUN apt-get -qq update && \
    apt-get -y install libgl1-mesa-dev xvfb libxrender1 build-essential mesa-common-dev mesa-utils freeglut3-dev ninja-build git cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ARG VTK_VERSION=v9.3.1
WORKDIR /home
RUN git clone --recursive https://gitlab.kitware.com/vtk/vtk.git -b ${VTK_VERSION} &&\
    cd vtk &&\
    cmake -GNinja -DVTK_WHEEL_BUILD=ON -DVTK_WRAP_PYTHON=ON -DVTK_GROUP_ENABLE_Web=WANT -B build -S . &&\
    cmake --build build/ &&\
    cd build/ &&\
    python setup.py bdist_wheel &&\
    cd .. &&\
    pip install build/dist/*.whl &&\
    rm -rf /home/vtk

ARG PYVISTA_VERSION=0.44.1
RUN pip install --no-cache-dir pyvista[jupyter]==${PYVISTA_VERSION} -v
