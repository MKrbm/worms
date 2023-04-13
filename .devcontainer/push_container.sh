# Usage: ./create_multi_arch_container.sh <username> <container_name> <architecture>

# Check that the required arguments are provided
if [ $# -ne 3 ]; then
  echo "Usage: ./create_multi_arch_container.sh <username> <container_name> <architecture>"
  exit 1
fi

username=$1
container_name=$2
architecture=$3

# Check that the architecture is either amd64 or arm64
if [ "$architecture" != "amd64" ] && [ "$architecture" != "arm64" ]; then
  echo "Error: Architecture must be either 'amd64' or 'arm64'"
  exit 1
fi

image_name="$username/$container_name:$architecture"
manifest_name="$username/$container_name:latest"

# Push the architecture-specific image to the registry
docker push "$image_name"

# If this is the first architecture being pushed, create the multi-architecture manifest
if [ "$(docker manifest inspect "$manifest_name" 2> /dev/null)" == "" ]; then
  docker manifest create "$manifest_name" "$image_name"
  docker manifest push "$manifest_name"
else
  # Otherwise, annotate the existing manifest with the new architecture
  docker manifest annotate "$manifest_name" "$image_name" --arch "$architecture"
  docker manifest push "$manifest_name"
fi