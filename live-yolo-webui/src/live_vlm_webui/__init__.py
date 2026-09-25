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

"""
Live VLM WebUI - Real-time Vision Language Model interaction web interface.

A universal web interface for streaming webcam feeds to Vision Language Models
with real-time AI analysis and system monitoring.
"""

__version__ = "0.4.0"
__author__ = "NVIDIA Corporation"
__license__ = "Apache-2.0"

from . import server
from . import video_processor
from . import gpu_monitor
from . import vlm_service

__all__ = ["server", "video_processor", "gpu_monitor", "vlm_service"]
