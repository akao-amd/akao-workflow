#!/bin/bash
set -e

echo "Installing GitHub CLI (gh) on Ubuntu 22.04..."

# Update package index
echo "Updating package index..."
apt update

# Install prerequisites
echo "Installing prerequisites..."
apt install -y curl gpg

# Add GitHub CLI repository
echo "Adding GitHub CLI repository..."
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null

# Install gh
echo "Installing gh..."
apt update
apt install -y gh

# Verify installation
echo "Verifying installation..."
gh --version

echo "GitHub CLI installed successfully!"
echo "You can authenticate with: gh auth login"
echo "Or use it without auth for public repos."
