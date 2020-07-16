import subprocess
import config.system as sys_config

for subject_id in range(2): # num_subjects
    subprocess.call(['python', sys_config.project_code_root + 'update_i2i.py', str(subject_id)])
