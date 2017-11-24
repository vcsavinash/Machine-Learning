import pandas as pd
import numpy as np
import sys

def processData(outDataframe,colData,colNumber,length):
	if ((length-1) == colNumber) and (colData.dtype == np.int64 or colData.dtype == np.float64):
		newColumn = (colData >= colData.mean())*1
		outDataframe[colNumber] = newColumn
	else:
		distinctValues = colData.unique()
		newColumn = colData
		numericValues = {}
		j = 0
		for val in distinctValues:
			numericValues[val] = j
			j = j+1
		new = []
		for index,row in colData.iteritems():
			new.insert(index,numericValues[row])
		outDataframe[colNumber] = new
		if (length-1) != colNumber:
			newColumn = (outDataframe[colNumber] - outDataframe[colNumber].mean())/outDataframe[colNumber].std()
		else:
			newColumn = (outDataframe[colNumber] - outDataframe[colNumber].min())/(outDataframe[colNumber].max()-outDataframe[colNumber].min())
		outDataframe[colNumber] = newColumn
	return

rawData = sys.argv[1]
processedData = sys.argv[2]

data = pd.read_table(rawData,sep='\t|,|:',index_col = False,header=None, engine = 'python')

outDataframe = pd.DataFrame()
data.replace(to_replace="[?]",value=np.nan,regex=True,inplace=True)
data = data.dropna()

for i in range(0,len(data.columns)):
	#if i!=len(data.columns)-2:	
	if len(data.columns)-1!=i and (data[i].dtype == np.int64 or data[i].dtype == np.float64):
		newColumn = ((data[i]-data[i].mean())/data[i].std())
		outDataframe[i]= newColumn
	else:
		processData(outDataframe,data[i],i,len(data.columns))


outDataframe.to_csv(processedData,sep=',',index = False)