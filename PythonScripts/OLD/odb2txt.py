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