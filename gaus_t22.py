import glob
import scipy.io.wavfile as wav
import numpy as np
#from python_speech_features import mfcc
import csv


#Conversion Force Aligment
conv={'a':'a','b':'b','d':'d','e':'e','f':'f','g':'g','i':'i','k':'k','l':'l','m':'m','n':'n','n~':'n~','o':'o','p':'p','r':'r','r(':'r(','s':'s','t':'t','tS':'tS','u':'u','x':'x','Z':'Z','a_7':'a','e_7':'e','i_7':'i','o_7':'o','u_7':'u','S_':'s'}

dic={'fon':[],'dur':[]}
dF={'a':[],'b':[],'d':[],'e':[],'f':[],'g':[],'i':[],'k':[],'l':[],'m':[],'n':[],'n~':[],'o':[],'p':[],'r':[],'r(':[],'s':[],'t':[],'tS':[],'u':[],'x':[],'Z':[]}

cont=0
for i,t22file in enumerate(glob.iglob('CorpusDimex100/*/T22/*/*.phn')):
	#print(t22file)    
	inicio="0.0"
	anterior="-"
	with open(t22file) as csvfile:
		reader=csv.reader(csvfile, delimiter=' ')
		for j,row in enumerate(reader):
			if row and row[0]!='MillisecondsPerFrame:' and row[0]!='END':
				try:
					if (row[2] == '.sil') or (row[2] == '.bn'):
						inicio=row[1]
						anterior="-"
					else:
						cont+=1            
						#row[0]=conv[row[0]]          
						dur=float(row[1])-float(inicio)
						#dur=float(row[2])-float(row[1])
						inicio=row[1]
						dic['fon'].append(row[2])
						dic['dur'].append(dur)
						dF[row[2]].append(anterior)
						anterior=row[2]
				except KeyError:
					print("KeyError: "+t22file)
				except:
					print("Index Error: "+t22file)
print(cont)


dic['fon']=np.array(dic['fon'])
dic['dur']=np.array(dic['dur'])

time_fon={}
med={}
std={}
time_fon['a']=dic['dur'][np.where(dic['fon']=='a')]
med['a']=time_fon['a'].mean()
std['a']=np.std(time_fon['a'])
time_fon['b']=dic['dur'][np.where(dic['fon']=='b')]
med['b']=time_fon['b'].mean()
std['b']=np.std(time_fon['b'])
time_fon['d']=dic['dur'][np.where(dic['fon']=='d')]
med['d']=time_fon['d'].mean()
std['d']=np.std(time_fon['d'])
time_fon['e']=dic['dur'][np.where(dic['fon']=='e')]
med['e']=time_fon['e'].mean()
std['e']=np.std(time_fon['e'])
time_fon['f']=dic['dur'][np.where(dic['fon']=='f')]
med['f']=time_fon['f'].mean()
std['f']=np.std(time_fon['f'])
time_fon['g']=dic['dur'][np.where(dic['fon']=='g')]
med['g']=time_fon['g'].mean()
std['g']=np.std(time_fon['g'])
time_fon['i']=dic['dur'][np.where(dic['fon']=='i')]
med['i']=time_fon['i'].mean()
std['i']=np.std(time_fon['i'])
time_fon['k']=dic['dur'][np.where(dic['fon']=='k')]
med['k']=time_fon['k'].mean()
std['k']=np.std(time_fon['k'])
time_fon['l']=dic['dur'][np.where(dic['fon']=='l')]
med['l']=time_fon['l'].mean()
std['l']=np.std(time_fon['l'])
time_fon['m']=dic['dur'][np.where(dic['fon']=='m')]
med['m']=time_fon['m'].mean()
std['m']=np.std(time_fon['m'])
time_fon['n']=dic['dur'][np.where(dic['fon']=='n')]
med['n']=time_fon['n'].mean()
std['n']=np.std(time_fon['n'])
time_fon['n~']=dic['dur'][np.where(dic['fon']=='n~')]
med['n~']=time_fon['n~'].mean()
std['n~']=np.std(time_fon['n~'])
time_fon['o']=dic['dur'][np.where(dic['fon']=='o')]
med['o']=time_fon['o'].mean()
std['o']=np.std(time_fon['o'])
time_fon['p']=dic['dur'][np.where(dic['fon']=='p')]
med['p']=time_fon['p'].mean()
std['p']=np.std(time_fon['p'])
time_fon['r']=dic['dur'][np.where(dic['fon']=='r')]
med['r']=time_fon['r'].mean()
std['r']=np.std(time_fon['r'])
time_fon['r(']=dic['dur'][np.where(dic['fon']=='r(')]
med['r(']=time_fon['r('].mean()
std['r(']=np.std(time_fon['r('])
time_fon['s']=dic['dur'][np.where(dic['fon']=='s')]
med['s']=time_fon['s'].mean()
std['s']=np.std(time_fon['s'])
time_fon['t']=dic['dur'][np.where(dic['fon']=='t')]
med['t']=time_fon['t'].mean()
std['t']=np.std(time_fon['t'])
time_fon['tS']=dic['dur'][np.where(dic['fon']=='tS')]
med['tS']=time_fon['tS'].mean()
std['tS']=np.std(time_fon['tS'])
time_fon['u']=dic['dur'][np.where(dic['fon']=='u')]
med['u']=time_fon['u'].mean()
std['u']=np.std(time_fon['u'])
time_fon['x']=dic['dur'][np.where(dic['fon']=='x')]
med['x']=time_fon['x'].mean()
std['x']=np.std(time_fon['x'])
time_fon['Z']=dic['dur'][np.where(dic['fon']=='Z')]
med['Z']=time_fon['Z'].mean()
std['Z']=np.std(time_fon['Z'])


np.save("Features/media.npy",med)
np.save("Features/std.npy",std)

maxi=0
l=''
LS=[]
globalMean=[]
globalStd=[]
for i in ['a','b','d','e','f','g','i','k','l','m','n','n~','o','p','r','r(','s','t','tS','u','x','Z']:
	print(i,med[i],std[i])    
	globalMean.append(med[i])   
	globalStd.append(std[i]) 
	durmax=med[i]+std[i]
	if durmax>maxi:
		maxi=durmax
		l=i
	LJ=[]
	for j in ['-','a','b','d','e','f','g','i','k','l','m','n','n~','o','p','r','r(','s','t','tS','u','x','Z']:
		LJ.append((np.array(dF[i])==j).sum())
	LS.append(LJ)
globalMean=np.array(globalMean)
globalStd=np.array(globalStd)
    
print(globalMean.mean(),globalStd.mean())
    
print(maxi)
print(l)

LS=np.array(LS)
np.save("Features/matDis.npy",LS)


				
						
						
