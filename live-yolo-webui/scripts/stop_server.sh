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

# Stop Live VLM WebUI Server

echo "Stopping Live VLM WebUI server..."
pkill -f "live_vlm_webui.server"

# Wait a moment
sleep 1

# Check if stopped
if pgrep -f "live_vlm_webui.server" > /dev/null; then
    echo "❌ Server still running, forcing kill..."
    pkill -9 -f "live_vlm_webui.server"
    sleep 1
fi

if ! pgrep -f "live_vlm_webui.server" > /dev/null; then
    echo "✓ Server stopped successfully"
else
    echo "❌ Failed to stop server"
    exit 1
fi
