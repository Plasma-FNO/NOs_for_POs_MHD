"""Render verified native bicubic density frames with fixed global color ranges."""
import sys,json,subprocess,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'code'))
sys.path.insert(0,'/pitagora_work/FUPB2_UKAEA_ML/vgopakum/runs/tear_circ_199_full_20261104/render_tools')
from native_fields import NativeMesh
import imageio_ffmpeg
encoder=imageio_ffmpeg.get_ffmpeg_exe()
parent=Path('/pitagora_scratch/userexternal/vgopakum/mastu_euler_ensemble100_20261105')
case=next(c for c in json.loads((root/'plan.json').read_text())['cases'] if c['case_id']==sys.argv[1])
assessment=json.loads((root/'cases'/f"{case['case_id']}.json").read_text())
if assessment['status']!='accepted':raise ValueError('Render accepted OOD cases only')
rate=assessment['selected_reference_rate']
folders=[parent/'runs'/f"{case['matched_id_case']}_fine"/'work',root/'runs'/f"{case['case_id']}_{rate}"/'work']
for folder in folders:assert json.loads((folder.parent/'result.json').read_text())['status']=='completed'
paths=[sorted(folder.glob('jorek??????.h5')) for folder in folders]
assert len(paths[0])==len(paths[1])==97
mesh=NativeMesh(paths[0][0]);polygons,basis,_=mesh.display_cells()
np.testing.assert_array_equal(mesh.read(paths[0][0])[0],mesh.read(paths[1][0])[0])
values=np.empty((97,2,len(polygons)),dtype='f8');times=[];sources=[];relative_density_changes=[]
for k,(p0,p1) in enumerate(zip(*paths)):
 q0,t0=mesh.read(p0);q1,t1=mesh.read(p1);assert abs(t1-t0)<1e-16
 for j,q in enumerate([q0,q1]):values[k,j]=mesh.evaluate(q[:,4:5],basis).ravel()
 relative_density_changes.append(float(mesh.rms(mesh.evaluate((q1-q0)[:,4:5]))[0]/mesh.rms(mesh.evaluate(q0[:,4:5]))[0]))
 times.append(t0);sources.append([{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [p0,p1]])
 if k%24==0:print('sampled',k,flush=True)
assert np.isfinite(values).all()
np.testing.assert_allclose(np.diff(times),np.diff(times)[0],rtol=1e-10,atol=1e-18)
lo,hi=float(values.min()),float(values.max());limit=float(np.abs(values[:,1]-values[:,0]).max())
out=root/'videos';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10})
labels=[f'Original · {case["matched_id_case"]}\nD = 1.000 × 10⁻⁵',f'OOD · {case["case_id"]}\nD = {case["coefficient_multiplier"]:.3f} × 10⁻⁵','Difference · OOD − original']
if 'gamma' in case['pde_coefficients']:
 labels=[f'Original · {case["matched_id_case"]}\nγ = 5/3; D = 10⁻⁵', f'OOD · {case["case_id"]}\nγ = {case["pde_coefficients"]["gamma"]:.3g}; D = 10⁻⁵', 'Difference · OOD − original']
records=[]
for name,indices in [(case['case_id']+'_density_comparison',[0,1,2])]:
 figsize=(12,7.2) if len(indices)==3 else (4.8,7.2)
 fig,axs=plt.subplots(1,len(indices),figsize=figsize,dpi=120,squeeze=False)
 fig.subplots_adjust(left=.08 if len(indices)==3 else .17,right=.97,bottom=.14,top=.78,wspace=.34)
 title=fig.suptitle('',fontsize=12,y=.97)
 artists=[]
 for ax,j in zip(axs.ravel(),indices):
  vals=values[0,j] if j<2 else values[0,1]-values[0,0]
  artist=PolyCollection(polygons,array=vals,cmap='viridis' if j<2 else 'RdBu_r',edgecolors='none',antialiaseds=False,rasterized=True)
  if j<2:artist.set_clim(lo,hi)
  else:artist.set_norm(TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
  ax.add_collection(artist);ax.set_aspect('equal');ax.set_xlim(.22,1.85);ax.set_ylim(-2.12,2.12)
  ax.set_xlabel('R [m]');ax.set_ylabel('Z [m]');ax.set_title(labels[j],fontsize=10);ax.set_facecolor('#f2f2f2')
  cbar=fig.colorbar(artist,ax=ax,orientation='horizontal',pad=.11,fraction=.035)
  cbar.set_label('Density ρ [simulation units]' if j<2 else 'Δρ [simulation units]',fontsize=9)
  artists.append(artist)
 fig.canvas.draw();width,height=fig.canvas.get_width_height();assert width%2==height%2==0
 partial=out/(name+'.partial.mp4');target=out/(name+'.mp4')
 command=[encoder,'-v','error','-y','-f','rawvideo','-pix_fmt','rgba','-s',f'{width}x{height}','-r','8','-i','-',
          '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)]
 with (out/(name+'.encoder.log')).open('wb') as log:
  process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
  try:
   for k,t in enumerate(times):
    for artist,j in zip(artists,indices):artist.set_array(values[k,j] if j<2 else values[k,1]-values[k,0])
    title.set_text(f'Native MAST-U density evolution · t = {t*1e6:.4f} μs\nElapsed = {(t-times[0])*1e6:.4f} μs · frame {k+1}/97')
    fig.canvas.draw();process.stdin.write(fig.canvas.buffer_rgba())
    if k in (0,96):fig.savefig(out/f'{name}_{"initial" if k==0 else "final"}.png',dpi=120)
   process.stdin.close();assert process.wait()==0
  finally:
   if process.poll() is None:process.kill();process.wait()
 plt.close(fig)
 verified=subprocess.run([encoder,'-v','error','-i',str(partial),'-map','0:v:0','-f','null','-','-progress','pipe:1','-nostats'],capture_output=True,text=True,check=True)
 decoded=[int(line.split('=')[1]) for line in verified.stdout.splitlines() if line.startswith('frame=')]
 assert decoded[-1]==97,(name,verified.stdout)
 partial.replace(target)
 records.append(dict(path=str(target),frames=97,fps=8,duration_seconds=97/8,decoded_frames=decoded[-1],sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
 print('verified',name,flush=True)
report=dict(original_case=case['matched_id_case'],ood_case=case['case_id'],coefficient_multiplier=case['coefficient_multiplier'],
 pde_coefficients=case['pde_coefficients'],identical_initial_states=True,times_seconds=times,field='rho',density_color_limits=[lo,hi],difference_color_limits=[-limit,limit],
 rendering='Native curved-element display subcells with bicubic center samples; no uniform-grid interpolation; fixed color limits over all 97 frames.',
 source_snapshots=sources,videos=records,selected_reference_rate=rate,
 density_relative_volume_rms_difference=relative_density_changes,
 final_density_relative_volume_rms_difference=relative_density_changes[-1],
 exceeds_ten_percent_density_change=bool(relative_density_changes[-1]>=.1))
(out/(case['case_id']+'_video_report.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(records,indent=2),flush=True)
