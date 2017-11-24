import pandas as pd
import numpy as np
import sys

def product(d1,d2):
	pr = np.dot(d1,d2)
	return pr
	
def forward(w,layers,train_data,t):
	x={}
	x[0] = train_data[t]
	for i in range(1,len(layers)):
		x[i-1] = np.append(1,x[i-1])
		x[i] = 1/(1+np.exp(-(product(x[i-1],w[i-1]))))
	return x
	
def backward(w,layers,train_output,x,t):	
	lr=0.01
	delta = {}
	deltaW = {}

	for h in range(len(layers)-1,-1,-1):
		if h==len(layers)-1:
			delta[len(layers)-1] = (train_output[t]-(x[len(layers)-1]))  * x[len(layers)-1] * (1-(x[len(layers)-1]))
		elif h+1 == len(layers)-1:
			delta[h] = x[h]*(1-x[h])*product(delta[h+1],w[h].transpose())
		else:
			delta[h] = x[h]*(1-x[h])*product(delta[h+1][1:],w[h].transpose())	
		if ((h+1 == len(layers)-1)and(h>0)):
			deltaW[h] = lr * product(x[h][:,None],delta[h+1][None,:])
		elif(h!=len(layers)-1):
			deltaW[h] = lr * product(x[h][:,None],delta[h+1][None,:][:,1:])
	return deltaW
	
def msd(obtained,target):
	sqrd = np.sum((target-obtained)**2)/len(target)
	return sqrd

if __name__=="__main__":
	data = np.genfromtxt(sys.argv[1],delimiter=',',skip_header = 1)
	numrows = data.shape[0]
	numcols = data.shape[1]
	num = int(numrows*int(sys.argv[2])/100.0)
	np.random.shuffle(data)
	train_data=data[0:num,0:numcols-1]
	test_data=data[num:,0:numcols-1]
	train_output=data[0:num,numcols-1:]
	test_output = data[num:,numcols-1:]

	layers = [numcols-1]
	hiddenLayersNum = int(sys.argv[4])
	for i in range(0, hiddenLayersNum):
		layers.append(int(sys.argv[5+i]))
	layers.append(1)

	w={}
	xval={}
	delW1={}
	for i in range(0,len(layers)-1):
		w[i] = np.random.random_sample((layers[i]+1,layers[i+1]))
	e = 0
	numOfIterations = int(sys.argv[3])
	for itr in range(0,numOfIterations):
		for t in range(0,len(train_data)-1):
			xval=forward(w,layers,train_data,t)
			delW1=backward(w,layers,train_output,xval,t)
			for k in range(0,len(layers)-1):
				w[k] = w[k] + delW1[k]
		for t in range(0,len(train_data)-1):
			outputpredict = forward(w,layers,train_data,t)
		err = msd(outputpredict[len(layers)-1],train_output)
		if err==0 or itr==numOfIterations:
			print (outputpredict)
			break
		e = err

	for t in range(0,len(test_data)-1):
		out = forward(w,layers,test_data,t)
	err1 = msd(out[len(layers)-1],test_output)

	for i in range(0,len(layers)-2):
		print ("Layer "+ str(i)+": ")
		for j in range(0,layers[i]+1):
			print ("\tNeuron"+ str(j)+ " weights : " + str(w[i][j]))
	print ("Layer "+str(len(layers)-2)+": ")
	for j in range(0, layers[len(layers) - 2]+1):
		print ("\tNeuron"+ str(j)+ " weights : " + str(w[len(layers) - 2][j]))

	print ("Total training error = "+ str(e))
	print ("Total test error = "+ str(err1))