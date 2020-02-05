import matlab.engine, os, glob

os.chdir("CompletedJobs")
OutputFiles=sorted(glob.glob('*.otp'), key=os.path.getsize)
os.chdir("..")

jobNames = [os.path.splitext(x)[0] for x in OutputFiles]

cwd = os.getcwd()

eng = matlab.engine.start_matlab()
eng.addpath(r'N:\My Drive\School\Masters\Nick Corbin\WIP\FRA Project\FRA-Matlab-Code',nargout=0)

for job in jobNames:
    # Check if the job has already been ran
    print(job)
    jobExtension = job.split('_', 1)[1]
    if not os.path.exists('MatlabFigures\dispersionCurve'+ jobExtension + '.png'):
        ret = eng.StressMeasurementFunction(job,cwd)
    else:
        print('Skipping job because it has already been processed')

