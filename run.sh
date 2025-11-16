#!/bin/bash
# Run script for AMD GPU (7900XTX with ROCm)
VK_LOADER_DEBUG=error VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json cargo run "$@"
