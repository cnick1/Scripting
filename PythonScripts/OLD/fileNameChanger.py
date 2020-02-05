import os
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for filename in files:
    os.rename(filename, filename.replace('SINE2CYCLE500HZ', 'RectangtularBeam_Aluminum6061_500Hz_p001mesh'))

   