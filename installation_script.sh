#!/bin/bash

# Update package lists
sudo apt update -y
sudo apt upgrade -y

# Install Firefox and Chromium
sudo apt install -y firefox chromium-browser

# Install nano & other useful packages
sudo apt install -y nano
sudo apt install tree

# Install UV python package manager
curl -sSL https://astral.sh/uv/install.sh | bash 
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc 


# Install Docker
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker

# Add current user to the docker group to allow running without sudo
sudo usermod -aG docker $USER

# Install Visual Studio Code
sudo apt install -y wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | sudo tee /usr/share/keyrings/packages.microsoft.gpg > /dev/null
echo "deb [arch=arm64 signed-by=/usr/share/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null
sudo apt update -y
sudo apt install -y code

# Give VS Code permissions to access Docker
sudo setfacl -m user:$USER:rw /var/run/docker.sock

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc


# Notify the user to restart the session to apply Docker group changes
echo "Please log out and log back in for Docker group changes to take effect."

echo "Setup complete!"