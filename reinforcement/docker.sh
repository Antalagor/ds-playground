ACTION=${1:-"restart"}

if [ "$ACTION" = "build" ]; then
  docker build -t rl-env .
fi

if [ "$ACTION" = "start" ]; then
  docker run --rm -d -e PYTHONPATH=/root/mount -v ${PWD}:/root/mount -p 8889:8888 -p 6006:6006 -t rl-env
fi

if [ "$ACTION" = "restart" ]; then
  docker kill "$(docker ps | grep "rl-env" | awk '{ print $1 }')"
  docker run --rm -d -e PYTHONPATH=/root/mount -v ${PWD}:/root/mount -p 8889:8888 -p 6006:6006 -t rl-env
fi

if [ "$ACTION" = "shell" ]; then
  docker exec -it "$(docker ps | grep "rl-env" | awk '{ print $1 }')" /bin/bash
fi
