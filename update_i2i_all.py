import subprocess
import config.system as sys_config

for subject_id in range(1, 2):
    # subprocess.call(['python', '/usr/bmicnas01/data-biwi-01/nkarani/projects/dg_seg/methods/tto_ss/tmp/update_i2i.py', str(subject_id)])
    subprocess.call(['python', sys_config.project_code_root + 'update_i2i.py', str(subject_id)])
