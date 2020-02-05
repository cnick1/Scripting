##################################################################### 
# Run with "abaqus python batchRun+odb2txt.py"
import os, glob, shutil
from odbAccess import *

InputFiles=sorted(glob.glob('*.inp'), key=os.path.getsize) # this will list all the input files in the folder, sorted by size

jobNames = [os.path.splitext(x)[0] for x in InputFiles] # Remove extensions

if not jobNames:
	sys.exit("No input files detected, execution aborted.")

# Make new working directory and copy input files
try:
    os.mkdir('WorkingDirectory')
except OSError:
    print ("Directory %s already exists" % 'WorkingDirectory')
else:
    print ("Successfully created the directory %s " % 'WorkingDirectory')

for InpFile in InputFiles:
	shutil.copy(InpFile, 'WorkingDirectory')

# Enter working directory and run jobs, then exit working directory
os.chdir("WorkingDirectory")

for job in jobNames: 
	print('INPUT FILE = %s' %job)
	str='abaqus job=%s int ask=off double cpus=4' %job 
	os.system(str) 
	
	######################## ODB2TXT ################################
	odb=openOdb(path=job+'.odb')
	WavePropagationStep = odb.steps['WavePropagation']
	
	sensor1 = WavePropagationStep.historyRegions['Node RECTANGULARBEAM-1.5']
	sensor2 = WavePropagationStep.historyRegions['Node RECTANGULARBEAM-1.6']
	sensor3 = WavePropagationStep.historyRegions['Node RECTANGULARBEAM-1.7']
	sensor4 = WavePropagationStep.historyRegions['Node RECTANGULARBEAM-1.8']
	
	time, sensor1u2 = zip(*sensor1.historyOutputs['U2'].data)
	sensor2u2 = zip(*sensor2.historyOutputs['U2'].data)[1]
	sensor3u2 = zip(*sensor3.historyOutputs['U2'].data)[1]
	sensor4u2 = zip(*sensor4.historyOutputs['U2'].data)[1]
	
	dispFile = open('../' + job + '.otp','w')
	for x in range(0, len(time)):
		dispFile.write('%10.6E   %10.6E   %10.6E   %10.6E   %10.6E\n' % (time[x], sensor1u2[x], sensor2u2[x], sensor3u2[x], sensor4u2[x]))
	dispFile.close()
	######################## ODB2TXT ################################

os.chdir('..')
##################################################################### 

OutputFiles=sorted(glob.glob('*.otp'), key=os.path.getsize) # this will list all the output files in the folder, sorted by size
completedJobNames = [os.path.splitext(x)[0] for x in OutputFiles] # Remove extensions

# Make new Already Ran directory and move input files
try:
    os.mkdir('CompletedJobs')
except OSError:
    print ("Directory %s already exists" % 'CompletedJobs')
else:
    print ("Successfully created the directory %s " % 'CompletedJobs')

for completedJobName in completedJobNames:
	try:
		if os.path.getsize(completedJobName + '.otp'):
			shutil.move(completedJobName + '.otp' , 'CompletedJobs')
			shutil.move(completedJobName + '.inp' , 'CompletedJobs')
		else:
			print('Sorry would not execute!')
	except OSError as o:
		# In case the file you asked does not exists!
		print('Sorry the file you asked does not exists!')
		print(str(o))
	

##################################################################### 