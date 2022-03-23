# Base image
FROM python:3.7-slim AS base-image

#===================================================================================================
# Base module builder
#===================================================================================================
FROM base-image AS builder

# Install the package dependencies
COPY requirements.txt /app/
RUN python -m venv /app/venv \
	&& ./app/venv/bin/python -m pip install --no-cache-dir -U pip \
	&& ./app/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

# Install the package
COPY MANIFEST.in pyproject.toml setup.cfg setup.py /app/
COPY src/ /app/src/
ARG GIT_DESCRIBE_STRING
RUN cd /app/ \
	&& ./venv/bin/pip install --no-cache-dir .

#===================================================================================================
# Production image
#===================================================================================================
FROM base-image

COPY --from=builder /app/venv/ /app/venv/

ENTRYPOINT ["/app/venv/bin/yellowgreen-multi"]
