#!/bin/bash
# Deploy Docker Image to Docker Hub
# Run this on master machine to make image available for workers

set -e

echo "=========================================="
echo "🐳 Deploy to Docker Hub"
echo "=========================================="
echo ""

# Get Docker Hub username
echo "Enter your Docker Hub username:"
read -r DOCKER_USERNAME

if [ -z "$DOCKER_USERNAME" ]; then
    echo "❌ Error: Username required"
    exit 1
fi

IMAGE_NAME="hetero-cluster-training"
DOCKER_TAG="$DOCKER_USERNAME/$IMAGE_NAME:latest"

echo ""
echo "Configuration:"
echo "  Local image: $IMAGE_NAME:latest"
echo "  Docker Hub: $DOCKER_TAG"
echo ""

# Check if image exists
if ! docker images | grep -q "$IMAGE_NAME"; then
    echo "❌ Error: Image '$IMAGE_NAME:latest' not found"
    echo ""
    echo "Build it first:"
    echo "  docker build -t $IMAGE_NAME:latest ."
    exit 1
fi

echo "✅ Image found"
echo ""

# Login to Docker Hub
echo "📝 Login to Docker Hub..."
echo "(Enter your Docker Hub password)"
if ! docker login; then
    echo "❌ Login failed"
    exit 1
fi

echo ""
echo "✅ Logged in successfully"
echo ""

# Tag the image
echo "🏷️  Tagging image..."
docker tag "$IMAGE_NAME:latest" "$DOCKER_TAG"
echo "✅ Tagged: $DOCKER_TAG"
echo ""

# Push to Docker Hub
echo "⬆️  Pushing to Docker Hub..."
echo "(This may take 3-5 minutes depending on your internet speed)"
echo ""

if docker push "$DOCKER_TAG"; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS! Image deployed to Docker Hub"
    echo "=========================================="
    echo ""
    echo "📦 Image available at:"
    echo "   $DOCKER_TAG"
    echo ""
    echo "📋 Share with your team:"
    echo "───────────────────────────────────────"
    echo "Docker Image: $DOCKER_TAG"
    echo "Master IP:    $(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | head -n1)"
    echo "Master Port:  29500"
    echo "───────────────────────────────────────"
    echo ""
    echo "🚀 Workers can now run:"
    echo "   docker pull $DOCKER_TAG"
    echo ""
    echo "See DOCKER_DEPLOYMENT_GUIDE.md for worker setup instructions"
    echo ""
else
    echo ""
    echo "❌ Push failed"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check internet connection"
    echo "  2. Verify Docker Hub credentials"
    echo "  3. Try logging in again: docker login"
    exit 1
fi
