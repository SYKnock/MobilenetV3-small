import sys
import time
import subprocess

if len(sys.argv) < 2:
    print("Usage: python power_logging.py [file name]")
    exit(0)

log = sys.argv[1]

f = open(log, 'w')

cnt = 1

while True:
    print(f"LOGGING------#{cnt}")
    
    output = subprocess.check_output("nvidia-smi -q -d power", shell=True, encoding='utf-8')
    f.write(output)
    f.write(f"!!!!!{cnt}\n")
    f.flush()
    cnt += 1
    time.sleep(2)
