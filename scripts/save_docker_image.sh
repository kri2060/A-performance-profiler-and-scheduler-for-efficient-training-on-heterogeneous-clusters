#!/bin/bash
# Save Docker Image to File (for USB/Network Transfer)

set -e

echo "=========================================="
echo "💾 Save Docker Image to File"
echo "=========================================="
echo ""

IMAGE_NAME="hetero-cluster-training:latest"
OUTPUT_FILE="hetero-cluster-training.tar"

# Check if image exists
if ! docker images | grep -q "hetero-cluster-training"; then
    echo "❌ Error: Image 'hetero-cluster-training:latest' not found"
    echo ""
    echo "Build it first:"
    echo "  docker build -t hetero-cluster-training:latest ."
    exit 1
fi

echo "✅ Image found: $IMAGE_NAME"
echo ""

# Get image size
IMAGE_SIZE=$(docker images hetero-cluster-training:latest --format "{{.Size}}")
echo "📦 Image size: $IMAGE_SIZE"
echo ""

# Check available space
AVAILABLE_SPACE=$(df -h . | tail -1 | awk '{print $4}')
echo "💽 Available space: $AVAILABLE_SPACE"
echo ""

# Ask for confirmation
echo "Output file: $(pwd)/$OUTPUT_FILE"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Save image
echo ""
echo "💾 Saving image to $OUTPUT_FILE..."
echo "(This may take 2-3 minutes)"
echo ""

if docker save $IMAGE_NAME -o $OUTPUT_FILE; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS! Image saved"
    echo "=========================================="
    echo ""

    # Show file info
    FILE_SIZE=$(ls -lh $OUTPUT_FILE | awk '{print $5}')
    echo "📁 File: $OUTPUT_FILE"
    echo "📊 Size: $FILE_SIZE"
    echo ""

    echo "📋 Next steps:"
    echo "───────────────────────────────────────"
    echo "1. Transfer to workers:"
    echo "   • USB: cp $OUTPUT_FILE /media/usb/"
    echo "   • SCP: scp $OUTPUT_FILE worker@IP:~/"
    echo ""
    echo "2. On worker machine:"
    echo "   docker load -i $OUTPUT_FILE"
    echo ""
    echo "3. Verify:"
    echo "   docker images | grep hetero-cluster-training"
    echo "───────────────────────────────────────"
    echo ""
    echo "See DOCKER_DEPLOYMENT_GUIDE.md for complete instructions"
    echo ""
else
    echo "❌ Failed to save image"
    exit 1
fi
