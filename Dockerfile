FROM continuumio/miniconda3

WORKDIR /app
ARG WORKSPACE=/app
ARG PROTO_FILE=t2v.proto

ARG USER=runner
ARG GROUP=runner-group
# ARG SRC_DIR=src

# Create non-privileged user to run
# RUN addgroup --system ${GROUP} && \
#    adduser --system --ingroup ${GROUP} ${USER} && \
#    chown -R ${USER}:${GROUP} ${WORKSPACE}

# Change to non-privileged user
# USER ${USER}

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ai4eu", "/bin/bash", "-c"]

# Make sure the environment is activated:
# RUN echo "Make sure flask is installed:"
# RUN python -c "import flask"

# Copy the all the stuff to the current workspace
COPY . ${WORKSPACE}/

# Build the grpc python files
COPY t2v.proto /
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ${PROTO_FILE}

# Expose port 8061 according to ai4eu specifications
EXPOSE 8061

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ai4eu", "python", "app.py"]

