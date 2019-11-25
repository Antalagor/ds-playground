ACTION=${1:-""}

if [ "$ACTION" = "build" ]
  then
    docker build -t rl-env .
  else
    docker run --rm  -e MOUNT=/root/mount -v ${PWD}:/root/mount -t -p 8888:8888 rl-env
fi
