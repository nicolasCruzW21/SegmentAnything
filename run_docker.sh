
#!/bin/bash

# Set the Docker image name
IMAGE_NAME="segv2"

# Set the container name (optional)
CONTAINER_NAME="segv2"

# Set any additional Docker run options (e.g., volume mounts, environment variables)
DOCKER_RUN_OPTIONS=""


# Run the Docker container
docker run -it --rm \
    --name "$CONTAINER_NAME" \
    $DOCKER_RUN_OPTIONS \
    $IMAGE_NAME \
    $COMMAND
