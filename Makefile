
DOCKER_BUILD = docker build --build-arg GIT_DESCRIBE_STRING=$$GIT_DESCRIBE_STRING -t $@ .
GIT_DESC_CMD = git describe --tags --long --dirty --match "*.*" 2>/dev/null || echo ''

# Module name, it must be the same name as the module command and a valid Docker name
MODULE_NAME = yellowgreen-multi


.PHONY: all build check module_name $(MODULE_NAME)

all: build check

# Build the production module
build: $(MODULE_NAME)

# Echo the module (repository) name
module_name:
	@echo $(MODULE_NAME)

# Echo the module version from the built Docker image
module_version:
	@docker run --rm $(MODULE_NAME) --version | sed 's/.* v//;s/+/-/'

# Define the module target
# Fallback to the null version if there was no Git tag
$(MODULE_NAME):
	@GIT_DESCRIBE_STRING=`$(GIT_DESC_CMD)` \
	&& echo "$(DOCKER_BUILD)" \
	&& $(DOCKER_BUILD)

# Rapid integrity check to make sure everything is correctly installed
check:
	docker run --rm --entrypoint bash $(MODULE_NAME) -c "\
		. /app/venv/bin/activate \
		&& $(MODULE_NAME) --version \
		&& exit \
		"
