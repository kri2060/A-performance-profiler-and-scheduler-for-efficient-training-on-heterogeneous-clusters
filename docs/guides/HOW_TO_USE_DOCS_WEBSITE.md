# 📚 How to Use the Documentation Website

Instead of reading through multiple markdown files, you can now view all the setup guides in a beautiful web interface!

---

## 🚀 Quick Start

### **Step 1: Launch the Documentation Server**

**On Linux/macOS:**
```bash
./launch_docs.sh
```

**On Windows:**
```cmd
launch_docs.bat
```

**Manual start (if scripts don't work):**
```bash
cd docs-website
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install Flask markdown
python3 app.py
```

### **Step 2: Open Your Browser**

Go to: **http://localhost:5000**

That's it! 🎉

---

## 📖 What You'll See

### **Beautiful Homepage**
- Navigation sidebar with all guides
- Quick links to get started
- Feature overview
- Nice gradient design

### **All Your Guides in One Place:**
- 🗺️ **Documentation Index** - Navigation hub
- ⚡ **Quick Start** - 5-minute setup
- ✅ **Setup Checklist** - Prerequisites
- 🎯 **Master Setup** - Master node configuration
- 🔧 **Worker Setup** - Worker node configuration
- 🚀 **Launch Guide** - Complete training examples
- 🎨 **Visual Diagrams** - Architecture and flows
- 🐳 **Docker Guide** - Advanced Docker setup

### **Features:**
- ✨ Syntax-highlighted code blocks
- 📊 Nicely rendered tables
- 🎨 Beautiful gradient design
- 📱 Mobile-friendly
- 🔍 Easy navigation
- 🎯 Direct links between guides

---

## 🌐 Access from Other Machines

The server runs on `0.0.0.0:5000`, which means it's accessible from any machine on your network!

**Your current network address:**
```
http://10.149.140.68:5000
```

**How to use this:**

1. **Keep the docs server running on your main machine**
2. **On worker machines**, open a browser and go to:
   ```
   http://YOUR_MAIN_MACHINE_IP:5000
   ```
3. **Follow the guides** right from the worker machine!

This is super useful when setting up workers - no need to copy files or switch machines!

---

## 💡 Usage Tips

### **Tip 1: Keep It Running While Setting Up**
Leave the documentation server running in a terminal while you set up your cluster. Access it from any machine to follow the guides.

### **Tip 2: Bookmark Common Pages**
Bookmark these for quick access:
- Quick Start: `http://localhost:5000/doc/quick-start`
- Master Setup: `http://localhost:5000/doc/master`
- Worker Setup: `http://localhost:5000/doc/worker`

### **Tip 3: Use Split Screen**
- Left side: Documentation website
- Right side: Terminal running commands

### **Tip 4: Access from Phone/Tablet**
The site is mobile-friendly! You can follow the guides on your phone while working on your laptop.

---

## 🛠️ Troubleshooting

### Issue: "Port 5000 already in use"

**Solution:** Change the port in `docs-website/app.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)  # Changed to 5001
```

### Issue: "Flask not found"

**Solution:** Install Flask:
```bash
pip install Flask markdown
```

### Issue: "Can't access from another machine"

**Solution:**
1. Check firewall - allow port 5000
2. Make sure both machines are on same network
3. Ping the main machine to verify connectivity

### Issue: Server crashes or errors

**Solution:**
```bash
# Stop any running instances
pkill -f "python.*app.py"

# Restart
./launch_docs.sh
```

---

## 🎨 What Makes This Better Than Reading Files?

### **Before (Reading .md files):**
- ❌ Hard to navigate between guides
- ❌ Plain text, hard to read
- ❌ Code blocks not highlighted
- ❌ Tables don't render well
- ❌ Have to open multiple files

### **After (Web interface):**
- ✅ Easy sidebar navigation
- ✅ Beautiful design, easy to read
- ✅ Syntax-highlighted code
- ✅ Tables render perfectly
- ✅ All guides in one place
- ✅ Mobile-friendly
- ✅ Can access from any machine

---

## 📱 Screenshots (What You'll See)

### Homepage:
```
┌─────────────────────────────────────────────┐
│  🚀 Multi-Node Distributed Training Setup  │
├──────────┬──────────────────────────────────┤
│          │  🎯 Welcome to Documentation    │
│ 📚 Docs  │                                  │
│          │  Quick Navigation:               │
│ 🗺️ Index │  1. Quick Start (5 Minutes)     │
│ ⚡ Quick  │  2. Setup Checklist             │
│ ✅ Check │  3. Master Setup                │
│ 🎯 Master│  4. Worker Setup                │
│ 🔧 Worker│  5. Launch Training             │
│ 🚀 Launch│                                  │
│ 🎨 Visual│  [All Guides Grid]              │
│ 🐳 Docker│                                  │
│          │  📚 Available Guides             │
│          │  [Guide Cards with Icons]       │
└──────────┴──────────────────────────────────┘
```

### Documentation Page:
```
┌─────────────────────────────────────────────┐
│  🚀 Multi-Node Distributed Training Setup  │
├──────────┬──────────────────────────────────┤
│          │  ⚡ Quick Start (5 Minutes)      │
│ Navigation│  ================================│
│ Sidebar  │                                  │
│ (Always  │  ## Step 1: Master Setup         │
│ Visible) │  ```bash                         │
│          │  export WORLD_SIZE=4             │
│          │  ./docker_run_master.sh          │
│          │  ```                             │
│          │                                  │
│          │  Expected Output:                │
│          │  Master IP: 192.168.1.100        │
│          │                                  │
│          │  [More content...]              │
└──────────┴──────────────────────────────────┘
```

---

## 🔒 Security Note

The documentation server is for **local network use only**. Don't expose it to the public internet.

- ✅ Safe on home/office network
- ✅ Safe on VPN
- ❌ Don't open port 5000 to internet

---

## 🎓 Recommended Workflow

**When setting up your cluster:**

1. **On your main machine:**
   ```bash
   ./launch_docs.sh
   ```

2. **Open browser to:** `http://localhost:5000`

3. **Follow the Quick Start guide**

4. **On worker machines:**
   - Open browser to: `http://MAIN_MACHINE_IP:5000`
   - Follow the Worker Setup guide
   - Keep it open for reference

5. **Switch between guides** using the sidebar as needed

6. **When done**, press Ctrl+C to stop the server

---

## 🎉 Enjoy Your Documentation!

The web interface makes setting up your cluster much easier. No more jumping between files!

**Quick Links:**
- Local: http://localhost:5000
- Network: http://10.149.140.68:5000

**Stop the server:** Press `Ctrl+C` in the terminal

**Restart anytime:** `./launch_docs.sh`

Happy cluster building! 🚀
