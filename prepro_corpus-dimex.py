
import glob
import scipy.io.wavfile as wav
import scipy.signal
import numpy as np
import csv
from python_speech_features import mfcc

media=np.load("Features/media.npy", allow_pickle=True)
media=media.item()
std=np.load("Features/std.npy", allow_pickle=True)
std=std.item()

#Conversion Force Aligment
conv={'a':'a','b':'b','d':'d','e':'e','f':'f','g':'g','i':'i','k':'k','l':'l','m':'m','n':'n','n~':'n~','o':'o','p':'p','r':'r','r(':'r(','s':'s','t':'t','tS':'tS','u':'u','x':'x','Z':'Z','a_7':'a','e_7':'e','i_7':'i','o_7':'o','u_7':'u','S_':'s'}


feat_X=[]
feat_Y=[]
antL=[]
cont=0
#Para cada archivo de marcas en el Corpus
for t22file in glob.iglob('CorpusDimex100/*/T22/*/*.phn',recursive=False):
	#Se obtiene el prefijo del archivo
	split=t22file.split("/")
	folder=split[1]
	subfolder=split[3]
	name=split[4]
	name_split=name.split(".")
	name=name_split[0]
	#Se busca por su correspondiente archivo de audio
	ls=glob.glob('CorpusDimex100/'+folder+'/audio_editado/'+subfolder+'/'+name+'.wav',recursive=False)
	try:
		audiofile=ls[0]
	except:
		print("ERROR: "+folder+name)
		continue
	#se lee el archivo de audio .wav
	try:
		sample_rate, signal = wav.read(audiofile)
	except:
		print("AUDIOFILE ERROR")
	#Se lee el archivo de marcas y se segmenta el archivo de audio original
	#print(t22file)
	inicio="0.0"
	with open(t22file) as csvfile:
		reader=csv.reader(csvfile, delimiter=' ')
		anterior='-'
		for i,row in enumerate(reader):
			if row and row[0]!='MillisecondsPerFrame:' and row[0]!='END':
			#Si la marca no es silencio o blanco
				try:
					if (row[2] == '.sil') or (row[2] == '.bn'):
						inicio=row[1]
						anterior='-'
					else:
						#row[0]=conv[row[0]]
						dur=float(row[1])-float(inicio)
						if ( dur>(media[row[2]]-std[row[2]]) ) and ( dur<(media[row[2]]+std[row[2]]) ):
							ns=signal[int(float(inicio)/1000 * sample_rate):int(float(row[1])/1000 * sample_rate)]
							if sample_rate!=16000 and len(ns)!=0:
								sampls=int(dur/1000*16000)
								#print('nuevo:'+str(sampls)+'orig:'+str(len(ns))+' '+str(sample_rate))                          
								ns=scipy.signal.resample(ns,sampls)
							try:                          
								#feat=ns
								feat=mfcc(ns,16000,numcep=26)
								print(feat.shape)                                
							except:
								print("_MFCC_ERROR_")                            
								continue                         
							#percen=np.percentile(feat,(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100),axis=0).T
							                            
							#print(percen[0,:]) 
							#feat=feat.T                            
							vector=feat.flatten()
							#print(vector[0:30])  
							#print(vector.shape)                        
							antL.append(anterior)
							feat_X.append(vector)
							feat_Y.append(row[2])
							cont+=1
							print(cont)
						inicio=row[1]
						anterior=row[2]

				except:
					print("Error "+t22file)                    
                    
print(cont)
np.save("Features/feat_X.npy",feat_X)
np.save("Features/feat_Y.npy",feat_Y)
np.save("Features/prevL.npy",antL)
	
	

	





