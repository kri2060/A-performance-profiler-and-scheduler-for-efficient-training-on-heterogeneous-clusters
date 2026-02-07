#!/bin/bash
set -e

echo "🐳 Detecting System..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    ID=$ID
else
    echo "❌ Underlying OS could not be detected."
    exit 1
fi

echo "Detected $OS ($ID)"

install_ubuntu() {
    echo "🐳 Installing Docker for Debian/Ubuntu..."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    
    echo \
      "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
      
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
}

install_arch() {
    echo "🐳 Installing Docker for Arch Linux..."
    # Update system and install docker + compose
    # docker-compose is now often part of docker-compose package or docker-buildx
    sudo pacman -Syu --noconfirm docker docker-compose
}

if [[ "$ID" == "ubuntu" || "$ID" == "debian" || "$ID" == "linuxmint" ]]; then
    install_ubuntu
elif [[ "$ID" == "arch" || "$ID" == "manjaro" || "$ID" == "endeavouros" ]]; then
    install_arch
else
    echo "❌ Unsupported Distribution: $ID"
    echo "Please install Docker manually: https://docs.docker.com/engine/install/"
    exit 1
fi

# Start Docker
echo "🐳 Starting Docker Service..."
sudo systemctl start docker.service
sudo systemctl enable docker.service

echo "🐳 Docker installed successfully!"

# Post-install steps
echo "🐳 Configuring user permissions (to run without sudo)..."
sudo usermod -aG docker $USER

echo "✅ Done! You may need to logout and log back in for group changes to take effect."
echo "👉 To verify, run: docker run hello-world"
