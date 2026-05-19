import math
import sys

LOG_FILE_PATH = "out.log"

# Define 256-color ANSI terminal codes
# 0.00 will be dim gray. Non-zero values scale from deep blue to bright red/white.
COLOR_ZERO = "\033[38;5;242m"  # Dim Gray
COLOR_RESET = "\033[0m"  # Reset formatting

# 10 distinct steps corresponding to:
# >0.0-0.1, >0.1-0.2, >0.2-0.3, >0.3-0.4, >0.4-0.5, >0.5-0.6, >0.6-0.7, >0.7-0.8, >0.8-0.9, >0.9-1.0
COLOR_STEPS = [
    "\033[38;5;39m",  # 0.1x - Light Blue
    "\033[38;5;43m",  # 0.2x - Cyan
    "\033[38;5;114m",  # 0.3x - Light Green
    "\033[38;5;148m",  # 0.4x - Yellow-Green
    "\033[38;5;220m",  # 0.5x - Yellow
    "\033[38;5;214m",  # 0.6x - Orange
    "\033[38;5;208m",  # 0.7x - Dark Orange
    "\033[38;5;202m",  # 0.8x - Red-Orange
    "\033[38;5;196m",  # 0.9x - Bright Red
    "\033[38;5;231m",  # 1.0x - Bright White
]


def get_colored_value(val_str):
    try:
        val = float(val_str)
    except ValueError:
        return val_str  # Return original string if it cannot be parsed

    # Exact zeros or structural padding
    if val == 0.0:
        return f"{COLOR_ZERO}{val_str}{COLOR_RESET}"

    # Safe math capping for values potentially out of bounds [0.01, 1.0]
    # Maps 0.01-0.10 -> index 0, 0.11-0.20 -> index 1, ..., up to 0.91-1.0+ -> index 9
    step_idx = math.ceil(val * 10) - 1
    step_idx = max(0, min(step_idx, 9))

    return f"{COLOR_STEPS[step_idx]}{val_str}{COLOR_RESET}"


def process_log(lines):
    output = []
    for line in lines:
        # Keep line breaks and whitespaces intact, split by commas
        parts = line.strip().split(",")
        colored_parts = []

        for part in parts:
            # Strip spaces to clean the string for conversion, but remember the spacing
            stripped = part.strip()
            if stripped:
                colored_val = get_colored_value(stripped)
                # Re-add structural padding space for clean grid alignment
                padding = " " * (len(part) - len(part.lstrip()))
                colored_parts.append(f"{padding}{colored_val}")
            else:
                colored_parts.append(part)

        # Print reconstructed line with comma separators preserved
        #print(",".join(colored_parts))
        output.append(",".join(colored_parts))
    return output


if __name__ == "__main__":
    try:
        with open(LOG_FILE_PATH, "r") as f:
            colored_output = process_log(f.readlines())
            for line in colored_output:
                print(line)

    except FileNotFoundError:
        print(
            f"Error: Could not find log file at '{LOG_FILE_PATH}'. Check the path variable.",
            file=sys.stderr,
        )
