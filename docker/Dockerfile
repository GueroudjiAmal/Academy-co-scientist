# ============================================================
# Academy Co-Scientist — PoC Execution Base Image
# ============================================================
#
# This is the BASE image used by DockerExecutorAgent.  It is
# never run directly for experiments.  Instead, DockerExecutorAgent
# inspects each generated script, detects extra package imports,
# and builds a thin DERIVED image on demand:
#
#   FROM academy-poc-runner
#   RUN pip install <extra-packages>
#
# Derived images are tagged "academy-poc-runner-ext-<hash8>" and
# cached locally by Docker, so identical dependency sets are built
# only once.  The experiment itself always runs with --network none;
# network access is used only during the image-build step.
#
# Build the base image:
#   docker build -t academy-poc-runner docker/
#
# Then set in configs/simulator_config.yaml:
#   poc:
#     docker_image: "academy-poc-runner"
#
# Packages listed here are the pre-baked "free" set — scripts that
# only use these libraries skip the derived-image build entirely.
# ============================================================

FROM python:3.11-slim

# System deps needed by some scientific libraries (e.g. matplotlib, h5py)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libhdf5-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Core scientific stack
RUN pip install --no-cache-dir \
    numpy>=1.26 \
    scipy>=1.13 \
    pandas>=2.2 \
    matplotlib>=3.9 \
    seaborn>=0.13 \
    scikit-learn>=1.5 \
    statsmodels>=0.14 \
    sympy>=1.13

# ML / gradient boosting
RUN pip install --no-cache-dir \
    xgboost>=2.0 \
    lightgbm>=4.3

# Data formats and I/O
RUN pip install --no-cache-dir \
    pyarrow>=16.0 \
    h5py>=3.11 \
    openpyxl>=3.1

# Graph and image utilities
RUN pip install --no-cache-dir \
    networkx>=3.3 \
    scikit-image>=0.23 \
    Pillow>=10.3

# Utilities
RUN pip install --no-cache-dir \
    tqdm>=4.66 \
    joblib>=1.4 \
    requests>=2.31 \
    tabulate>=0.9

WORKDIR /experiment

# Code is passed via stdin at runtime:
#   docker run --rm -i academy-poc-runner python -
# No COPY or CMD needed for the stdin execution mode.
