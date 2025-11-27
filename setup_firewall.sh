#!/bin/bash
# Setup firewall for distributed training
# Run this on the MASTER node with sudo

echo "=========================================="
echo "Firewall Setup for Distributed Training"
echo "=========================================="

PORT=29500

echo "Setting up firewall to allow port $PORT..."

# Check which firewall system is in use
if command -v firewall-cmd &> /dev/null; then
    echo "Using firewalld..."
    sudo firewall-cmd --permanent --add-port=$PORT/tcp
    sudo firewall-cmd --reload
    echo "✓ Port $PORT opened in firewalld"

elif command -v ufw &> /dev/null; then
    echo "Using ufw..."
    sudo ufw allow $PORT/tcp
    echo "✓ Port $PORT opened in ufw"

else
    echo "Using iptables..."
    # Check if rule already exists
    if sudo iptables -C INPUT -p tcp --dport $PORT -j ACCEPT 2>/dev/null; then
        echo "✓ Port $PORT already allowed"
    else
        sudo iptables -I INPUT -p tcp --dport $PORT -j ACCEPT
        echo "✓ Port $PORT opened in iptables"

        # Try to save rules (depends on distro)
        if command -v iptables-save &> /dev/null; then
            echo "Saving iptables rules..."
            sudo iptables-save > /tmp/iptables.rules
            echo "Rules saved to /tmp/iptables.rules"
            echo "Note: Rules may not persist after reboot on some systems"
        fi
    fi
fi

echo ""
echo "Verifying port $PORT is open..."
if sudo iptables -L INPUT -n | grep -q "$PORT"; then
    echo "✓ Port $PORT is OPEN in iptables"
else
    echo "⚠ Could not verify port $PORT"
fi

echo ""
echo "=========================================="
echo "Testing from worker: nc -zv 10.161.199.69 $PORT"
echo "=========================================="
