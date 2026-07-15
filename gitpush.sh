#!/usr/bin/env bash

set -euo pipefail

# Usage:
# ./git_push.sh "model=qwen2.5-14b temperature=0"

commit_message="${1:?Usage: $0 \"commit message / hyperparameters\"}"
env_file="${ENV_FILE:-.env}"

if [[ ! -f "$env_file" ]]; then
    echo "Error: $env_file was not found."
    exit 1
fi

# Load variables from .env and export them
set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

if [[ -z "${Github_Username:-}" ]]; then
    echo "Error: Github_Username is missing from $env_file."
    exit 1
fi

if [[ -z "${Github_Token:-}" ]]; then
    echo "Error: Github_Token is missing from $env_file."
    exit 1
fi

git add .

if git diff --cached --quiet; then
    echo "No staged changes to commit."
else
    git commit -m "$commit_message"
fi

ASKPASS_SCRIPT="$(mktemp)"

cleanup() {
    rm -f "$ASKPASS_SCRIPT"
    unset Github_Username Github_Token
}

trap cleanup EXIT

cat > "$ASKPASS_SCRIPT" <<'EOF'
#!/usr/bin/env bash

case "$1" in
    *Username*)
        printf '%s\n' "$Github_Username"
        ;;
    *Password*)
        printf '%s\n' "$Github_Token"
        ;;
    *)
        exit 1
        ;;
esac
EOF

chmod 700 "$ASKPASS_SCRIPT"

export Github_Username
export Github_Token

env \
    -u SSH_ASKPASS \
    -u VSCODE_GIT_ASKPASS_NODE \
    -u VSCODE_GIT_ASKPASS_MAIN \
    -u VSCODE_GIT_ASKPASS_EXTRA_ARGS \
    GIT_ASKPASS="$ASKPASS_SCRIPT" \
    GIT_TERMINAL_PROMPT=0 \
    git push -u origin main

echo "Commit and push completed successfully."