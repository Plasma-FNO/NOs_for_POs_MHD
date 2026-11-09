from pathlib import Path
import numpy as np,json
b=Path('/pitagora_work/FUPB2_UKAEA_ML/vgopakum/mastu-opsplit/cpu-development-20261102/native_bridge_20261103/enabled8')
seen={};duplicates=[];files=[]
for p in sorted(b.glob('opsplit-native-rank-*.bin')):
 h=np.fromfile(p,dtype='<i4',count=8);q=int(h[1]);nfloat=6+74*q*q+32+112+112*112;record=33*4+nfloat*8
 assert (p.stat().st_size-32)%record==0
 n=(p.stat().st_size-32)//record
 m=np.memmap(p,dtype='u1',mode='r')
 ids=np.ndarray((n,),dtype='<i4',buffer=m,offset=32,strides=(record,))
 files.append({'path':str(p),'records':n,'minimum_id':int(ids.min()),'maximum_id':int(ids.max()),'unique':len(np.unique(ids))})
 for i,e in enumerate(ids):
  if int(e) in seen:
   previous,oldi=seen[int(e)];old=np.memmap(previous,dtype='u1',mode='r')
   a=np.asarray(old[32+oldi*record:32+(oldi+1)*record]);c=np.asarray(m[32+i*record:32+(i+1)*record])
   duplicates.append({'element':int(e),'first_rank_file':previous.name,'second_rank_file':p.name,'all_record_bytes_identical':bool(np.array_equal(a,c))})
  else:seen[int(e)]=(p,i)
print(json.dumps({'files':files,'unique_elements':len(seen),'duplicates':duplicates},indent=2))
