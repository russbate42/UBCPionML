import subprocess as sub

cmd = ['python','run.py']

result = sub.run(cmd, stdout=sub.PIPE, stderr=sub.STDOUT)
content = str(result.stdout).split('\\n')

with open('run.log','w') as f:
    for item in content:
        f.write("%s\n" % item)
