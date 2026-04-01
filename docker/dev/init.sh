export DOCKER_UID=$(id -u)
export DOCKER_GID=$(id -g)
docker compose -f docker-compose.yml build --no-cache
docker compose -f docker-compose.yml up -d
