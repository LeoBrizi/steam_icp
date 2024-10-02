#!/bin/sh

docker run -it --rm --name steam_icp \
  --privileged \
  --gpus all \
  --network=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ${HOME}:${HOME}:rw \
  steam_icp
