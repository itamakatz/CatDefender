import subprocess
state = "1"
# prog = subprocess.Popen(['runas', '/noprofile', '/user:Administrator', 'NeedsAdminPrivilege.exe', state],stdin=subprocess.PIPE)
# prog = subprocess.Popen(['./toggle_LED.sh', '1'],stdin=subprocess.PIPE)
subprocess.run(["./toggle_LED.sh", '1'])
# prog.stdin.write('3.14'.encode())
# prog.communicate()