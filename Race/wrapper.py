import os 
import subprocess
import time
import pickle 

# map = "t3" # t1_triple, t2_triple, t3, t4, shanghai_intl_circuit
map = "shanghai_intl_circuit"
# map = "t1_triple"
# map = "t2_triple"
# map = "t3"
# map = "t4"


controller_process = subprocess.Popen("python3 automatic_control_GRAIC.py --sync -m {}".format(map), shell=True)
time.sleep(5)
scenario_process = subprocess.Popen("python3 scenario.py -m {}".format(map), shell=True, stdout=None)
