##################################################################### 
##################### RailStressMeasurement.py ######################
#####################################################################
# This script takes over from where batchRun.py finishes, and where 
# Matlab takes over This includes the following:
#	0) Import packages and define functions
# 	1) Enter CompletedJobs directory and get Abaqus job output list
# 	2) Start Matlab engine and iterate through every file, processing 
#       the results and saving the output plots

#	0) Import packages and define functions
import matlab.engine, os, glob

# 	1) Enter CompletedJobs directory and get Abaqus job output list
os.chdir("CompletedJobs")
OutputFiles=sorted(glob.glob('*.otp'), key=os.path.getsize)
os.chdir("..")

jobNames = [os.path.splitext(x)[0] for x in OutputFiles]

cwd = os.getcwd()

# 	2) Start Matlab engine and iterate through every file, processing 
#       the results and saving the output plots
eng = matlab.engine.start_matlab()
eng.addpath(r'N:\My Drive\School\Masters\Nick Corbin\WIP\FRA Project\FRA-Matlab-Code',nargout=0)

for job in jobNames:
    # Check if the job has already been ran
    print(job)
    jobExtension = job.split('_', 3)[3]
    if not os.path.exists('MatlabFigures\dispersionCurve'+ jobExtension + '.png'):
        ret = eng.StressMeasurementFunction(job,cwd)
    else:
        print('Skipping job because it has already been processed')

