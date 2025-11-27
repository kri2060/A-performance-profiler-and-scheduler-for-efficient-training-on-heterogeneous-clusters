# WSL2 Network Configuration for Distributed Training

## Problem
WSL2 uses NAT networking which isolates it from the host network. Direct communication between Linux and WSL2 for distributed training fails.

## Solution Options

### Option 1: WSL2 Mirrored Networking (Windows 11 only - RECOMMENDED)

1. Create/edit `C:\Users\<YourUsername>\.wslconfig`:
```ini
[wsl2]
networkingMode=mirrored
```

2. Restart WSL2:
```cmd
wsl --shutdown
```

3. Start WSL2 again and verify IP:
```bash
ip addr show eth0
```

Your WSL2 will now have the same IP as Windows host (10.161.199.174).

4. Run training normally - no port forwarding needed!

### Option 2: Port Forwarding (All Windows versions)

**Already configured in START_WORKER.sh**

1. On Windows PowerShell (as Administrator):
```powershell
# Forward port from Windows host to WSL2
netsh interface portproxy add v4tov4 listenport=29500 listenaddress=10.161.199.174 connectport=29500 connectaddress=172.18.16.1

# Also forward a range for Gloo's dynamic ports
netsh interface portproxy add v4tov4 listenport=40000-41000 listenaddress=10.161.199.174 connectport=40000-41000 connectaddress=172.18.16.1

# Allow firewall
netsh advfirewall firewall add rule name="Distributed Training" dir=in action=allow protocol=TCP localport=29500
netsh advfirewall firewall add rule name="Gloo Dynamic Ports" dir=in action=allow protocol=TCP localport=40000-41000
```

2. Verify port forwarding:
```powershell
netsh interface portproxy show all
```

3. Test from Linux master:
```bash
nc -zv 10.161.199.174 29500
```

### Option 3: Run Python Directly on Windows (Not WSL2)

Install Python + CUDA on Windows directly and run the training script natively.

## Current Configuration

- Linux Master: 10.161.199.69
- Windows Host: 10.161.199.174  
- WSL2 Internal: 172.18.16.1

## Debugging

If still failing, check:
1. Can master ping worker? `ping 10.161.199.174`
2. Can worker ping master? `ping 10.161.199.69`
3. Is port forwarding active? `netsh interface portproxy show all`
4. Is firewall allowing? `netsh advfirewall firewall show rule name="Distributed Training"`
