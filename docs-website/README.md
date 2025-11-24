# Documentation Website

A beautiful web interface to view all multi-node setup guides.

## Quick Start

### Linux/macOS:
```bash
./launch_docs.sh
```

### Windows:
```cmd
launch_docs.bat
```

Then open your browser to: **http://localhost:5000**

## Features

- 📚 All setup guides in one place
- 🎨 Beautiful, responsive design
- 🔍 Easy navigation sidebar
- 📱 Mobile-friendly
- 🎯 Syntax-highlighted code blocks
- 📊 Tables and diagrams rendered nicely

## Available Guides

- ⚡ Quick Start (5 Minutes)
- ✅ Setup Checklist
- 🎯 Master Node Setup
- 🔧 Worker Node Setup
- 🚀 Launch Guide & Examples
- 🎨 Visual Setup Diagrams
- 🐳 Docker Multi-Node Guide
- 🗺️ Documentation Index

## Manual Installation

If the launch script doesn't work:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

## Access from Other Machines

The server binds to `0.0.0.0:5000`, so you can access it from other machines on your network:

```
http://YOUR_IP:5000
```

Replace `YOUR_IP` with the IP address of the machine running the server.

This is useful when setting up worker machines - keep the docs server running on your main machine and access it from workers!

## Technologies

- **Flask** - Web framework
- **Markdown** - Document format
- **Python-Markdown** - Markdown to HTML conversion
- Pure CSS - No frameworks, lightweight and fast
