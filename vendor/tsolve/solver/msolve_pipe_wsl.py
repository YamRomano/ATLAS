import sys
import os
import subprocess

MSOLVE_BINARY = "/mnt/c/Users/user/Desktop/PnP_Project/OptimalPnP-main/msolve"

def main():
    msolve_input = sys.stdin.read()

    if not msolve_input:
        return

    mem_file = f"/dev/shm/msolve_tmp_{os.getpid()}.ms"

    try:
        with open(mem_file, "w") as f:
            f.write(msolve_input)

        res = subprocess.run(
            [MSOLVE_BINARY, "-f", mem_file, "-v", "0"],
            capture_output=True,
            text=True
        )

        out_txt = res.stdout
        sys.stdout.write(out_txt)
        sys.stdout.flush()

    finally:
        if os.path.exists(mem_file):
            os.remove(mem_file)


if __name__ == "__main__":
    main()