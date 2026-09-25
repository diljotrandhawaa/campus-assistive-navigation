#!/bin/bash
# Example commands for different use cases of Live VLM WebUI

# ============================================================================
# Scene Description (Default)
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --prompt "Describe what you see in this image in one sentence."

# ============================================================================
# Object Detection
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --prompt "List all objects visible in this image, separated by commas."

# ============================================================================
# Safety Monitoring
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --prompt "Are there any safety hazards visible? Answer with 'ALERT: <description>' or 'SAFE'."

# ============================================================================
# Activity Recognition
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --prompt "What is the person doing? Describe their activity briefly."

# ============================================================================
# Accessibility - Detailed Description
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --prompt "Provide a detailed description of the scene for a visually impaired person."

# ============================================================================
# People Counting
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --prompt "How many people are in this image? Just state the number and their general locations."

# ============================================================================
# Emotion Detection
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --prompt "Describe the facial expressions and emotional states of people visible."

# ============================================================================
# Color Analysis
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --prompt "What are the dominant colors in this image?"

# ============================================================================
# Text Reading (OCR)
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --prompt "Read and transcribe any text visible in the image."

# ============================================================================
# Posture/Ergonomics Check
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --prompt "Analyze the person's posture. Is it ergonomically correct? Give brief feedback."

# ============================================================================
# Pet Detection
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --prompt "Are there any pets or animals visible? If yes, describe them."

# ============================================================================
# Fashion/Outfit Description
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --prompt "Describe the clothing and fashion items visible in the image."

# ============================================================================
# Using with Ollama (easier setup)
# ============================================================================
# python server.py --model llava:7b \
#   --api-base http://localhost:11434/v1 \
#   --prompt "Describe what you see in this image in one sentence."

# ============================================================================
# Using with SGLang
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:30000/v1 \
#   --prompt "Describe what you see in this image in one sentence."

# ============================================================================
# High Frequency Updates (more responsive)
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --process-every 15 \
#   --prompt "Describe what you see in one sentence."

# ============================================================================
# Low Frequency Updates (less resource intensive)
# ============================================================================
# python server.py --model llama-3.2-11b-vision-instruct \
#   --api-base http://localhost:8000/v1 \
#   --process-every 60 \
#   --prompt "Describe what you see in one sentence."


