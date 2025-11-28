#!/bin/bash
#
# Get IP Address Helper Script
# Displays the IP address that should be used for MASTER_ADDR
#

echo "=========================================="
echo "IP Address Detection"
echo "=========================================="

# Detect network interface
NETWORK_INTERFACE=""
if ip link show wlan0 &>/dev/null; then
    NETWORK_INTERFACE="wlan0"
elif ip link show wlp &>/dev/null; then
    NETWORK_INTERFACE=$(ip link show | grep -o "wlp[^:]*" | head -n1)
elif ip link show eth0 &>/dev/null; then
    NETWORK_INTERFACE="eth0"
elif ip link show enp &>/dev/null; then
    NETWORK_INTERFACE=$(ip link show | grep -o "enp[^:]*" | head -n1)
fi

if [ -n "$NETWORK_INTERFACE" ]; then
    IP_ADDRESS=$(ip addr show $NETWORK_INTERFACE | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}' | cut -d/ -f1 | head -n1)

    echo "Network Interface: $NETWORK_INTERFACE"
    echo "IP Address: $IP_ADDRESS"
    echo ""
    echo "Use this IP as MASTER_ADDR on worker nodes:"
    echo "  export MASTER_ADDR=$IP_ADDRESS"
    echo ""
    echo "Or edit START_WORKER.sh line 83:"
    echo "  MASTER_ADDR=$IP_ADDRESS"
else
    echo "ERROR: Could not detect network interface"
    echo ""
    echo "Available interfaces:"
    ip link show | grep -E "^[0-9]+:" | cut -d: -f2 | tr -d ' '
    echo ""
    echo "Manually get IP with:"
    echo "  ip addr show <interface-name>"
fi

echo "=========================================="
