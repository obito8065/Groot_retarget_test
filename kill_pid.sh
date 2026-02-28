#!/bin/bash 

for pid in {1314515..1321508}; do
    kill -9 $pid 2>/dev/null
done