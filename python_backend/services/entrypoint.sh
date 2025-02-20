#!/bin/bash
# entrypoint.sh

for mod in $(pactl list short modules | grep Virtual_Sink | awk '{print $1}'); do
    pactl unload-module "$mod"
done

# Load the virtual sink module.
pactl load-module module-null-sink sink_name=Virtual_Sink sink_properties=device.description=Virtual_Sink

# Set the virtual sink monitor as the default source.
pactl set-default-source Virtual_Sink.monitor

# Execute the CMD from the Dockerfile.
exec "$@"
