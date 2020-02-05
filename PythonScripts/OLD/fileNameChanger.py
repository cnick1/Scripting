import os
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for filename in files:
    os.rename(filename, filename.replace('50', '500'))

   