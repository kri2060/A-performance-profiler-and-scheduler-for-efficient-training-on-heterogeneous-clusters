#!/bin/bash
# Get current master IP address

MASTER_IP=$(ip addr show wlan0 | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}' | cut -d/ -f1 | tail -n1)

echo "=========================================="
echo "Current Master IP: $MASTER_IP"
echo "=========================================="
echo ""
echo "Update worker with:"
echo "  On worker machine, edit START_WORKER.sh"
echo "  Change: export MASTER_ADDR=$MASTER_IP"
echo ""
echo "Or run on worker:"
echo "  sed -i 's/export MASTER_ADDR=.*/export MASTER_ADDR=$MASTER_IP/' START_WORKER.sh"
echo "=========================================="
