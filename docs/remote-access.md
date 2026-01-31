# Remote Access Guide

Control your robot from anywhere — not just your local network.

## Option 1: Tailscale (Recommended for most users)

**Best for:** Personal use, easy setup, works through any NAT/firewall.

```bash
# Install on BOTH brain and body:
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Get your Tailscale IP:
tailscale ip -4

# Update your config:
export PIDOG_HOST="100.x.x.x"  # Tailscale IP of the body
```

**Pros:** Zero-config, encrypted, works everywhere, free for personal use (up to 100 devices)
**Cons:** Requires Tailscale account, slight latency overhead

## Option 2: WireGuard (Best performance)

**Best for:** Production setups, lowest latency, full control.

```bash
# Install
sudo apt install wireguard

# Generate keys (on each device)
wg genkey | tee privatekey | wg pubkey > publickey

# Configure (see example configs below)
sudo cp wg0.conf /etc/wireguard/
sudo systemctl enable --now wg-quick@wg0
```

**Pros:** Kernel-level performance, tiny overhead, battle-tested
**Cons:** Manual key management, needs a server with public IP or port forwarding

## Option 3: Cloudflare Tunnel (Public access)

**Best for:** Demos, public APIs, webhook integrations.

```bash
# Install cloudflared
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo apt-key add -
sudo apt install cloudflared

# Login and create tunnel
cloudflared tunnel login
cloudflared tunnel create pidog
cloudflared tunnel route dns pidog pidog.yourdomain.com

# Run
cloudflared tunnel run pidog
```

## Option 4: Telegram Bot (Simplest)

No VPN needed — control via Telegram messages.

```bash
# 1. Create a bot with @BotFather on Telegram
# 2. Set your token:
export TELEGRAM_BOT_TOKEN="your-token"
export TELEGRAM_ALLOWED_USERS="your-user-id"

# 3. Start the bot:
python3 brain/telegram_bot.py
```

Commands: `/status`, `/photo`, `/speak`, `/move`, `/voice`, `/face`, `/battery`

## Combining Approaches

For the best experience, use **Tailscale + Telegram**:
- Tailscale for full API access and low-latency control
- Telegram for quick commands when you're on your phone

## Security Checklist

- [ ] Set `NOX_API_TOKEN` on the body
- [ ] Restrict Telegram bot to your user ID(s)
- [ ] Use Tailscale ACLs to limit device access
- [ ] Enable firewall: only expose port 8888 to VPN interface
- [ ] Rotate API tokens periodically
