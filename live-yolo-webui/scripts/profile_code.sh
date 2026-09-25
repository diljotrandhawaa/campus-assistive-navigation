#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Profile specific components for performance analysis

set -e

BLUE='\033[0;34m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Code Profiling Tool${NC}"
echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo ""

# Check if component is specified
if [ $# -eq 0 ]; then
    echo "Usage: $0 <component> [options]"
    echo ""
    echo "Components:"
    echo "  video_processor  - Profile video processing functions"
    echo "  vlm_service      - Profile VLM service"
    echo "  server           - Profile server endpoints"
    echo "  all              - Profile all performance tests"
    echo ""
    echo "Options:"
    echo "  --output FILE    - Save profile to file (default: profile.stats)"
    echo "  --visualize      - Generate visualization (requires snakeviz)"
    echo ""
    exit 1
fi

COMPONENT=$1
OUTPUT_FILE="profile_${COMPONENT}.stats"
VISUALIZE=false

shift

while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --visualize)
            VISUALIZE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Profiling: ${COMPONENT}${NC}"
echo ""

case $COMPONENT in
    video_processor)
        TEST_PATH="tests/unit/test_video_processor.py::TestVideoProcessorPerformance"
        ;;
    vlm_service)
        TEST_PATH="tests/unit/test_vlm_service.py -m performance"
        ;;
    server)
        TEST_PATH="tests/integration/test_server.py -m performance"
        ;;
    all)
        TEST_PATH="tests/ -m performance"
        ;;
    *)
        echo "Unknown component: $COMPONENT"
        exit 1
        ;;
esac

# Run profiling
echo "Running profiler..."
python -m cProfile -o "$OUTPUT_FILE" -m pytest $TEST_PATH -v

echo ""
echo -e "${GREEN}✅ Profiling complete!${NC}"
echo ""
echo "Profile saved to: $OUTPUT_FILE"
echo ""
echo "Analyze with:"
echo "  python -m pstats $OUTPUT_FILE"
echo ""

# Generate text report
echo "Top 20 time-consuming functions:"
echo "-----------------------------------"
python << EOF
import pstats
p = pstats.Stats('$OUTPUT_FILE')
p.sort_stats('cumulative')
p.print_stats(20)
EOF

# Visualize if requested
if [ "$VISUALIZE" = true ]; then
    if command -v snakeviz &> /dev/null; then
        echo ""
        echo -e "${YELLOW}Opening visualization...${NC}"
        snakeviz "$OUTPUT_FILE"
    else
        echo ""
        echo -e "${YELLOW}Install snakeviz for visualization:${NC}"
        echo "  pip install snakeviz"
    fi
fi
