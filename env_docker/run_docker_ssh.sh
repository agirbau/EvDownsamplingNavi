#!/bin/bash

# Add your Docker commands here
docker rm evdownsamplingnavi
docker run \
    --name evdownsamplingnavi \
    -it \
    -v /home/andreu/work/projects/research/methods/EvDownsamplingNavi:/home/user/app \
    -v /home/andreu/datasets:/home/user/datasets \
    -v /petaco:/petaco \
    -v /home/andreu/work/andreu_utils:/home/user/global_utils \
    evdownsamplingnavi
