#!/bin/bash
# ML-Lab deploy script
cd /home/pi/ML-Lab
pkill -f "python.*app.py" 2>/dev/null
sleep 1
> /tmp/ml-lab.log
nohup python3 app.py > /tmp/ml-lab.log 2>&1 &
echo "PID: $!"
sleep 3
tail -10 /tmp/ml-lab.log
