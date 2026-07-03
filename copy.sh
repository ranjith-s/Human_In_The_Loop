#!/bin/bash

# --- Configuration ---
SOURCE_DIR="/home/naman/Downloads/HITL_Lab-main/"  # Change this to your local path

# The script will check these paths in order and use the first one it finds.
# If none exist, it defaults to the user's home folder.
POSSIBLE_DESTS=("/Data/" "/DATA/" "/data/")
DEFAULT_DEST="~/ " 

IP_START=101
IP_END=140
USERNAMES=("teaching" "teaching1")
PASSWORDS=("ds123" "dslab123")

# --- Execution ---
echo "Starting bulk SCP transfer with destination checking..."

for i in $(seq $IP_START $IP_END); do
    IP="172.18.40.$i"
    echo "----------------------------------------"
    echo "Targeting System: $IP"
    echo "----------------------------------------"
    
    # Quick ping check to see if host is online
    if ! ping -c 1 -W 1 "$IP" > /dev/null 2>&1; then
        echo "Host $IP is unreachable. Skipping..."
        continue
    fi

    SUCCESS=false

    for USER in "${USERNAMES[@]}"; do
        for PASS in "${PASSWORDS[@]}"; do
            echo "Testing credentials for $USER @ $IP..."

            # Test SSH connection and check which directory exists
            # This loop runs a remote '[ -d /path ]' command over SSH
            REMOTE_DEST=""
            for DEST in "${POSSIBLE_DESTS[@]}"; do
                if sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 "${USER}@${IP}" "[ -d $DEST ]" 2>/dev/null; then
                    REMOTE_DEST=$DEST
                    echo "Found valid directory: $REMOTE_DEST"
                    break
                fi
            done

            # If none of the specific data folders were found, but the SSH credentials worked,
            # we default to the home directory so the transfer doesn't completely fail.
            if [ -z "$REMOTE_DEST" ]; then
                # Quick sanity check to see if the password actually works before defaulting
                if sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=2 "${USER}@${IP}" "true" 2>/dev/null; then
                    REMOTE_DEST=$DEFAULT_DEST
                    echo "⚠️ Data folders not found. Defaulting to home directory ($REMOTE_DEST)"
                else
                    # Wrong credentials, move to the next password combo
                    continue
                fi
            fi

            # If we reached here, credentials are good and REMOTE_DEST is set. Proceed to copy.
            echo "Copying files to ${REMOTE_DEST}..."
            sshpass -p "$PASS" scp -r -o StrictHostKeyChecking=no "${SOURCE_DIR}" "${USER}@${IP}:${REMOTE_DEST}"

            if [ $? -eq 0 ]; then
                echo "✅ Success: Copied to $IP ($USER) inside $REMOTE_DEST"
                SUCCESS=true
                break 2 # Break out of credential loops, move to next IP
            fi
        done
    done

    if [ "$SUCCESS" = false ]; then
        echo "❌ Failed to copy to $IP (Credentials failed or host rejected connection)"
    fi
done

echo "----------------------------------------"
echo "Bulk SCP process completed."
