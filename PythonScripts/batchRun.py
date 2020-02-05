##################################################################### 
# Run with "abaqus python batchRun+odb2txt.py"
import os, glob, shutil, threading, sys
from odbAccess import *

def odb2txt(job):
	odb=openOdb(path=job+'.odb')
	WavePropagationStep = odb.steps['WavePropagation']
	sensorNodes = WavePropagationStep.historyRegions.keys()
	sensorDatas=[WavePropagationStep.historyRegions[sensor] for sensor in sensorNodes]
	dataOutput=sensorDatas[1].historyOutputs.keys()[0]
	time = zip(*sensorDatas[1].historyOutputs[dataOutput].data)[0]
	sensorDisplacements = [zip(*sensorData.historyOutputs[dataOutput].data)[1] for sensorData in sensorDatas]
	dispFile = open('../' + job + '.otp','w')
	for x in range(0, len(time)):
		dispFile.write('%10.6E   %10.6E   %10.6E   %10.6E   %10.6E\n' % (time[x], sensorDisplacements[0][x], sensorDisplacements[1][x], sensorDisplacements[2][x], sensorDisplacements[3][x]))
	dispFile.close()
	return

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
	print('INPUT FILE = %s \n' %job)
	str='abaqus job=%s int ask=off double cpus=4' %job
	os.system(str)
	odb_thread = threading.Thread(target=odb2txt, args=[job])
	odb_thread.daemon = True
	odb_thread.start()

odb_thread.join()

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

