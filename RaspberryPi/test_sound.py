import subprocess
result = subprocess.run(
    ["aplay", "-D", "hw:2", "./ding.wav"], capture_output=True, text=True
)