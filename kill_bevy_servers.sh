#!/bin/bash
# Kill all headless Bevy server instances

pkill -f "bevy_3d_dodge --headless" && echo "Killed all Bevy server instances" || echo "No Bevy servers running"
