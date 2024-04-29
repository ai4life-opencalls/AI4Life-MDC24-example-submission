FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

# GPU
#RUN python -m pip install torch==1.13.1+cu116 \
#    torchvision==0.14.1+cu116  \
#    --extra-index-url https://download.pytorch.org/whl/cu116
#
## CPU
#RUN python -m pip install torch  \
#    torchvision --index-url https://download.pytorch.org/whl/cpu

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user resources /opt/app/resources
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user src /opt/app/src

RUN  ["python", "src/create_model.py"]
ENTRYPOINT ["python", "inference.py"]
