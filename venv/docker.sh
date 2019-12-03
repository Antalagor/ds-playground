ACTION=${1:-""}

if [ "$ACTION" = "build" ]
  then
    docker build -t rl-env .
fi

if [ "$ACTION" = "restart" ]
  then
    docker kill "$(docker ps | grep "rl-env" | awk '{ print $1 }')"
    docker run --rm -d -e PYTHONPATH=/root/mount -v ${PWD}:/root/mount -t -p 8889:8888 bidder-rl-env
fi
