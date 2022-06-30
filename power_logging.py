import os
import time
import subprocess

f = open("power_log.txt", 'w')

cnt = 1

while True:
    print(f"LOGGING------#{cnt}")
    
    output = subprocess.check_output("nvidia-smi -q -d power", shell=True, encoding='utf-8')
    f.write(output)
    f.write(f"!!!!!{cnt}\n")
    
    cnt += 1
    time.sleep(2)
