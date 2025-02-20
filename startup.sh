#!/bin/bash

pactl load-module module-null-sink sink_name=Virtual_Sink sink_properties=device.description=Virtual_Sink.monitor &&
pactl set-default-source Virtual_Sink.monitor 
docker compose up --build