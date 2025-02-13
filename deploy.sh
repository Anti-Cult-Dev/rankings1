#!/bin/bash

# Remote server details
SERVER="65.109.188.143"
USER="root"
REMOTE_DIR="/root/token-monitor"

# Create deployment package
echo "Creating deployment package..."
tar czf deploy.tar.gz \
    cli.py \
    monitor.py \
    liveness_agent.py \
    requirements.txt \
    DEPLOYMENT.md \
    templates/

# Copy files to server
echo "Copying files to server..."
scp deploy.tar.gz $USER@$SERVER:$REMOTE_DIR.tar.gz

# Execute remote commands
ssh $USER@$SERVER << 'EOF'
    # Create directory and extract files
    mkdir -p /root/token-monitor
    cd /root/token-monitor
    tar xzf ../token-monitor.tar.gz

    # Setup Python environment
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    # Make CLI executable
    chmod +x cli.py

    # Create systemd service
    cat > /etc/systemd/system/token-monitor.service << 'EOL'
[Unit]
Description=Token Monitor Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/token-monitor
Environment=PATH=/root/token-monitor/venv/bin
ExecStart=/root/token-monitor/venv/bin/python cli.py start monitor
Restart=always

[Install]
WantedBy=multi-user.target
EOL

    # Reload systemd and start service
    systemctl daemon-reload
    systemctl enable token-monitor
    systemctl start token-monitor

    # Cleanup
    rm ../token-monitor.tar.gz
    echo "Deployment complete!"
EOF
