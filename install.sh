#!/bin/bash
#
# GridPythia – Service Installation Script
#
# Usage: sudo ./install.sh [OPTIONS]
#
# Options:
#   -H, --host    HOST   Bind address for the web UI  (default: 0.0.0.0)
#   -p, --port    PORT   Web UI TCP port              (default: 8080)
#   -c, --config  PATH   Path to config.yaml          (default: <install-dir>/config.yaml)
#   -h, --help           Show this help message and exit
#
# Examples:
#   sudo ./install.sh
#   sudo ./install.sh --host 0.0.0.0 --port 8080
#   sudo ./install.sh --config /etc/gridpythia/config.yaml
#

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
err()  { echo -e "${RED}✗ ERROR:${NC} $*" >&2; }

# ── Constants ─────────────────────────────────────────────────────────────────
SERVICE_NAME="gridpythia"
SERVICE_FILE="${SERVICE_NAME}.service"
SERVICE_USER="gridpythia"
SERVICE_GROUP="pythia"

# ── Defaults ──────────────────────────────────────────────────────────────────
HOST="0.0.0.0"
PORT="8080"
CONFIG_PATH=""   # resolved to <SCRIPT_DIR>/config.yaml below

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -25
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -H|--host)    HOST="$2";        shift 2 ;;
        -p|--port)    PORT="$2";        shift 2 ;;
        -c|--config)  CONFIG_PATH="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *) err "Unknown option: $1"; echo "Run with --help for usage."; exit 1 ;;
    esac
done

# ── Must run as root (via sudo) ───────────────────────────────────────────────
echo "======================================================================="
echo "  GridPythia – Service Installation"
echo "======================================================================="

if [[ "$EUID" -ne 0 ]]; then
    err "This script must be run with sudo."
    echo "  Usage: sudo ./install.sh [OPTIONS]"
    exit 1
fi

if [[ -z "${SUDO_USER:-}" ]]; then
    err "Could not determine the invoking user. Run via sudo, not as root directly."
    exit 1
fi

REAL_USER="$SUDO_USER"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve config path (default: config.yaml inside project directory)
if [[ -z "$CONFIG_PATH" ]]; then
    CONFIG_PATH="${SCRIPT_DIR}/config.yaml"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    err "Config file not found: $CONFIG_PATH"
    echo "  Provide --config <path> or ensure config.yaml exists in the project directory."
    exit 1
fi

# ── Ensure uv is available ────────────────────────────────────────────────────
echo ""
echo "======================================================================="
echo "  Checking for uv"
echo "======================================================================="

UV_BIN="$(command -v uv 2>/dev/null || true)"
if [[ -z "$UV_BIN" ]]; then
    warn "uv not found – installing via official installer to /usr/local/bin …"
    if ! command -v curl &>/dev/null; then
        err "curl is required to install uv. Run: apt-get install curl"
        exit 1
    fi
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR=/usr/local/bin sh
    UV_BIN="$(command -v uv 2>/dev/null || true)"
    if [[ -z "$UV_BIN" ]]; then
        err "uv installation failed. Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
    ok "uv installed: $UV_BIN ($("$UV_BIN" --version))"
else
    ok "uv found: $UV_BIN ($("$UV_BIN" --version))"
fi

# ── Create venv if missing ────────────────────────────────────────────────────
if [[ ! -d "${SCRIPT_DIR}/.venv" ]]; then
    warn ".venv not found – creating virtual environment and installing dependencies …"
    sudo -u "$REAL_USER" "$UV_BIN" venv "${SCRIPT_DIR}/.venv"
    sudo -u "$REAL_USER" "$UV_BIN" sync --no-dev --project "${SCRIPT_DIR}"
    ok "Virtual environment created and dependencies installed"
fi

# ── Locate Python ─────────────────────────────────────────────────────────────
PYTHON_BIN=""
for candidate in \
    "${SCRIPT_DIR}/.venv/bin/python3" \
    "${SCRIPT_DIR}/venv/bin/python3" \
    "$(command -v python3 2>/dev/null || true)"; do
    if [[ -x "$candidate" ]]; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    err "Python3 executable not found."
    exit 1
fi

PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1)
EXTRA_ARGS="--host ${HOST} --port ${PORT} --config ${CONFIG_PATH}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "  Install directory   : $SCRIPT_DIR"
echo "  Service user/group  : $SERVICE_USER / $SERVICE_GROUP"
echo "  Installing user     : $REAL_USER"
echo "  Python              : $PYTHON_VERSION"
echo "                        $PYTHON_BIN"
echo "  Bind address        : $HOST:$PORT"
echo "  Config file         : $CONFIG_PATH"
echo ""

read -rp "Continue with installation? (y/N) " REPLY
echo
[[ "$REPLY" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

# ── Create group and service user ─────────────────────────────────────────────
echo ""
echo "======================================================================="
echo "  Setting up user and group"
echo "======================================================================="

if ! getent group "$SERVICE_GROUP" &>/dev/null; then
    groupadd --system "$SERVICE_GROUP"
    ok "Created system group: $SERVICE_GROUP"
else
    warn "Group '$SERVICE_GROUP' already exists – skipping"
fi

if ! getent passwd "$SERVICE_USER" &>/dev/null; then
    useradd \
        --system \
        --gid "$SERVICE_GROUP" \
        --no-create-home \
        --shell /usr/sbin/nologin \
        --comment "GridPythia service account" \
        "$SERVICE_USER"
    ok "Created system user: $SERVICE_USER (no login, no home)"
else
    warn "User '$SERVICE_USER' already exists – skipping"
fi

# Add the installer to the pythia group so they can still edit project files
if ! id -nG "$REAL_USER" 2>/dev/null | grep -qw "$SERVICE_GROUP"; then
    usermod -aG "$SERVICE_GROUP" "$REAL_USER"
    ok "Added $REAL_USER to group $SERVICE_GROUP"
    warn "You must log out and back in for the group change to take effect."
else
    warn "$REAL_USER is already a member of $SERVICE_GROUP"
fi

# ── File ownership and permissions ────────────────────────────────────────────
echo ""
echo "======================================================================="
echo "  Setting file ownership and permissions"
echo "======================================================================="

chown -R "${SERVICE_USER}:${SERVICE_GROUP}" "$SCRIPT_DIR"
ok "Ownership set to ${SERVICE_USER}:${SERVICE_GROUP}"

# Directories: rwxrwsr-x (setgid so new files inherit the group)
find "$SCRIPT_DIR" -type d -exec chmod 2775 {} \;
ok "Directories: 2775 (rwxrwsr-x)"

# Files: use capital X – preserves existing execute bits (venv bins, .so libs)
chmod -R u=rwX,g=rwX,o=rX "$SCRIPT_DIR"
ok "Files: owner/group rw, others r (execute bits preserved)"

# Ensure install/uninstall scripts are executable
[[ -f "$SCRIPT_DIR/install.sh" ]]   && chmod 775 "$SCRIPT_DIR/install.sh"
[[ -f "$SCRIPT_DIR/uninstall.sh" ]] && chmod 775 "$SCRIPT_DIR/uninstall.sh"

# If config lives outside the project dir, grant service user read access
if [[ "$CONFIG_PATH" != "$SCRIPT_DIR"* ]]; then
    chown "${SERVICE_USER}:${SERVICE_GROUP}" "$CONFIG_PATH"
    chmod 640 "$CONFIG_PATH"
    ok "Config file ownership set: $CONFIG_PATH"
fi

# Allow git operations in this directory for the real user
sudo -u "$REAL_USER" git -C "$SCRIPT_DIR" config --local safe.directory "$SCRIPT_DIR" 2>/dev/null \
    && ok "git safe.directory configured for $REAL_USER" \
    || warn "git safe.directory skipped (not a git repo?)"

# ── Install systemd service ───────────────────────────────────────────────────
echo ""
echo "======================================================================="
echo "  Installing systemd service"
echo "======================================================================="

if [[ ! -f "$SCRIPT_DIR/$SERVICE_FILE" ]]; then
    err "Service template not found: $SCRIPT_DIR/$SERVICE_FILE"
    exit 1
fi

# Stop and disable any existing installation
if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    systemctl stop "$SERVICE_NAME"
    warn "Stopped existing running service"
fi
if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
    systemctl disable "$SERVICE_NAME"
fi

# Generate concrete service file from template
TMP_SERVICE="$(mktemp /tmp/${SERVICE_NAME}.XXXXXX.service)"
sed \
    -e "s|{{WORKING_DIR}}|${SCRIPT_DIR}|g" \
    -e "s|{{PYTHON_BIN}}|${PYTHON_BIN}|g" \
    -e "s|{{EXTRA_ARGS}}|${EXTRA_ARGS}|g" \
    "$SCRIPT_DIR/$SERVICE_FILE" > "$TMP_SERVICE"

install -m 644 -o root -g root "$TMP_SERVICE" "/etc/systemd/system/${SERVICE_FILE}"
rm -f "$TMP_SERVICE"
ok "Installed /etc/systemd/system/${SERVICE_FILE}"

systemctl daemon-reload
ok "systemd daemon reloaded"

systemctl enable "$SERVICE_NAME"
ok "Service enabled (auto-start on boot)"

systemctl start "$SERVICE_NAME"
sleep 2

# ── Result ────────────────────────────────────────────────────────────────────
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo ""
    echo "======================================================================="
    echo -e "  ${GREEN}Installation successful!${NC}"
    echo "======================================================================="
    echo ""
    echo "  Web UI:  http://${HOST}:${PORT}/"
    echo ""
    echo "  Useful commands:"
    echo "    sudo systemctl status  $SERVICE_NAME"
    echo "    sudo journalctl -u $SERVICE_NAME -f"
    echo "    sudo systemctl restart $SERVICE_NAME"
    echo "    sudo systemctl stop    $SERVICE_NAME"
    echo "    sudo systemctl disable $SERVICE_NAME"
    echo ""
    echo "  IMPORTANT: Log out and back in so the '$SERVICE_GROUP' group takes effect"
    echo "  (required to edit project files as $REAL_USER)."
    echo ""
    echo "  Recent log output:"
    echo "  -------------------------------------------------------------------"
    journalctl -u "$SERVICE_NAME" -n 15 --no-pager
    echo "  -------------------------------------------------------------------"
else
    echo ""
    echo "======================================================================="
    echo -e "  ${RED}Service failed to start!${NC}"
    echo "======================================================================="
    echo ""
    echo "  Full logs:"
    journalctl -u "$SERVICE_NAME" -n 40 --no-pager
    echo ""
    echo "  Common causes:"
    echo "    - venv missing or incomplete  →  uv sync --no-dev"
    echo "    - Missing solver library      →  apt-get install libatomic1  (on ARM)"
    echo "    - Config error in $CONFIG_PATH"
    echo ""
    exit 1
fi
