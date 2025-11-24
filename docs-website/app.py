#!/usr/bin/env python3
"""
Multi-Node Setup Documentation Website
A simple Flask app to serve the setup guides in a nice web interface
"""

from flask import Flask, render_template, send_from_directory
import markdown
import os
from pathlib import Path

app = Flask(__name__)

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Documentation files
DOCS = {
    'index': {
        'title': 'Multi-Node Setup Documentation',
        'file': 'MULTINODE_INDEX.md',
        'icon': '🗺️',
        'description': 'Complete navigation guide'
    },
    'quick-start': {
        'title': 'Quick Start (5 Minutes)',
        'file': 'QUICK_START_MULTINODE.md',
        'icon': '⚡',
        'description': 'Get started in 5 minutes'
    },
    'checklist': {
        'title': 'Setup Checklist',
        'file': 'SETUP_CHECKLIST.md',
        'icon': '✅',
        'description': 'Prerequisites and requirements'
    },
    'master': {
        'title': 'Master Node Setup',
        'file': 'MASTER_SETUP.md',
        'icon': '🎯',
        'description': 'Setup the master node (RANK=0)'
    },
    'worker': {
        'title': 'Worker Node Setup',
        'file': 'WORKER_SETUP.md',
        'icon': '🔧',
        'description': 'Setup worker nodes (RANK 1, 2, 3...)'
    },
    'launch': {
        'title': 'Launch Guide',
        'file': 'LAUNCH_GUIDE.md',
        'icon': '🚀',
        'description': 'Complete training examples'
    },
    'diagrams': {
        'title': 'Visual Setup Guide',
        'file': 'SETUP_DIAGRAM.md',
        'icon': '🎨',
        'description': 'Architecture diagrams and flows'
    },
    'docker': {
        'title': 'Docker Multi-Node Guide',
        'file': 'DOCKER_MULTINODE_SETUP.md',
        'icon': '🐳',
        'description': 'Advanced Docker setup'
    },
}

def read_markdown(filename):
    """Read and convert markdown file to HTML"""
    filepath = BASE_DIR / filename
    if not filepath.exists():
        return f"<p>File not found: {filename}</p>"

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Convert markdown to HTML with extensions
    html = markdown.markdown(
        content,
        extensions=[
            'markdown.extensions.fenced_code',
            'markdown.extensions.tables',
            'markdown.extensions.toc',
            'markdown.extensions.codehilite',
            'markdown.extensions.nl2br',
        ]
    )
    return html

@app.route('/')
def home():
    """Main page with navigation"""
    return render_template('home.html', docs=DOCS)

@app.route('/doc/<doc_id>')
def show_doc(doc_id):
    """Display a specific documentation page"""
    if doc_id not in DOCS:
        return "Documentation not found", 404

    doc = DOCS[doc_id]
    content = read_markdown(doc['file'])

    return render_template('doc.html',
                         title=doc['title'],
                         content=content,
                         docs=DOCS,
                         current=doc_id)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Multi-Node Setup Documentation Server")
    print("=" * 60)
    print("\n📖 Access the documentation at:")
    print("   http://localhost:5000")
    print("\n📚 Available guides:")
    for doc_id, doc in DOCS.items():
        print(f"   {doc['icon']} {doc['title']}")
    print("\n" + "=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
