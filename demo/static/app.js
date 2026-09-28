'use strict';
const $ = id => document.getElementById(id);
const state = {token:null,ready:false,csv:null,fileName:'',example:null,result:null,target:'Hip_Mom',examples:[],busy:false};
const colors = {pelvis:'#8894a1',hip:'#147e87',knee:'#489ba6',ankle:'#94c2c4',pred:'#ba761e',ref:'#34464f',band:'#8b979d'};
const targetLabels = {Hip_Mom:'Hip moment',Knee_Mom:'Knee moment',Ankle_Mom:'Ankle moment',GRF_V:'Vertical ground reaction force'};
const configLabels = {full:'Pelvis + Hip + Knee + Ankle',no_pelvis:'Hip + Knee + Ankle',no_hip_ang:'Pelvis + Knee + Ankle',no_knee_ang:'Pelvis + Hip + Ankle',no_ankle_ang:'Pelvis + Hip + Knee',ladder_no_ankle_pelvis:'Hip + Knee',ladder_hip_only:'Hip'};
const hiddenGroups = {full:[],no_pelvis:['Pelvis'],no_hip_ang:['Hip'],no_knee_ang:['Knee'],no_ankle_ang:['Ankle'],ladder_no_ankle_pelvis:['Ankle','Pelvis'],ladder_hip_only:['Ankle','Pelvis','Knee']};
const cohortLabels = {normal:'Typically developing',cp:'Cerebral palsy',vdk_stroke:'After stroke',bmclab_pd:'Parkinson’s disease'};
const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function note(text,error=false){$('run-status').textContent=text;$('run-status').classList.toggle('error',error);}
function updateRun(){ $('run').disabled=!state.ready||!state.csv||!$('confirm').checked||state.busy; }
function csvCell(v){const s=String(v??'');return /[",\n\r]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}
function download(name,text,type='application/json'){const a=document.createElement('a');const u=URL.createObjectURL(new Blob([text],{type}));a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
function legend(id,items){$(id).innerHTML=items.map(v=>`<span><i class="swatch ${v.band?'band':''}" style="background:${v.color}"></i>${esc(v.label)}</span>`).join('');}
function chart(id,series,band=null){
  const W=720,H=180,m={l:46,r:14,t:10,b:37},w=W-m.l-m.r,h=H-m.t-m.b;
  let vals=series.flatMap(s=>s.y).filter(Number.isFinite);
  if(band) vals.push(...band.lo.filter(Number.isFinite),...band.hi.filter(Number.isFinite));
  if(!vals.length){$(id).innerHTML='<p class="subtle">No observed angles for this limb and configuration.</p>';return;}
  let low=Math.min(...vals),high=Math.max(...vals),range=high-low||1;low-=range*.12;high+=range*.12;
  const x=i=>m.l+i/100*w,y=v=>m.t+(high-v)/(high-low)*h;
  const path=arr=>arr.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(' ');
  const fmt=v=>Math.abs(v)>=10?v.toFixed(0):v.toFixed(2).replace(/0+$/,'').replace(/\.$/,'');
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Waveforms over a normalised gait cycle"><rect width="${W}" height="${H}" fill="white"/>`;
  for(let i=0;i<4;i++){const v=low+(high-low)*i/3;svg+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="#e9eeef"/><text x="${m.l-9}" y="${y(v)+3}" text-anchor="end" fill="#71838d" font-size="10">${fmt(v)}</text>`;}
  for(const i of [0,25,50,75,100])svg+=`<text x="${x(i)}" y="${H-20}" text-anchor="middle" fill="#71838d" font-size="10">${i}</text>`;
  svg+=`<text x="${m.l+w/2}" y="${H-3}" text-anchor="middle" fill="#71838d" font-size="10">Gait cycle (%)</text>`;
  if(band){const d=path(band.hi)+' '+band.lo.map((_,k)=>{const i=band.lo.length-1-k;return `L${x(i).toFixed(2)},${y(band.lo[i]).toFixed(2)}`;}).join(' ')+' Z';svg+=`<path d="${d}" fill="${colors.band}" opacity=".2"/>`;}
  for(const s of series)svg+=`<path d="${path(s.y)}" fill="none" stroke="${s.color}" stroke-width="${s.width||2}" stroke-linecap="round" stroke-linejoin="round"/>`;
  svg+='</svg>';$(id).innerHTML=svg;
}
function meanAngles(){
  if(state.example){const input=state.example.input;return Object.fromEntries(input.channelNames.map((name,i)=>[name,input.mean.map(row=>row[i])]).filter(([,a])=>a.every(Number.isFinite)));}
  return state.result?.mean_angles||{};
}
function draw(){
  if(!state.example&&!state.result)return;
  $('empty').hidden=true;$('plots').hidden=false;$('download').disabled=false;
  const ex=state.example,config=$('input-set').value,side=ex?ex.side:$('side').value;
  $('side').value=side;$('side').disabled=!!ex;
  for(const b of document.querySelectorAll('[data-target]')){b.disabled=!!ex&&!ex.targets[b.dataset.target]||(!ex&&state.result.cohort==='bmclab_pd'&&b.dataset.target!=='GRF_V');b.classList.toggle('active',b.dataset.target===state.target);}
  if(ex){
    $('download-metadata').hidden=true;
    $('result-kind').textContent=ex.targets[state.target].configurations[config]?.provenance?.rerun?'Held-out example · replayed':'Held-out example';$('result-title').textContent=cohortLabels[ex.cohort]||ex.label;
    $('result-description').textContent=`${ex.nCycles} cycles · ${configLabels[config]}`;
    $('example-input').hidden=false;$('example-input').href=`/api/examples/${ex.id}.csv`;$('example-input').download=ex.id+'_angles.csv';
  }else{
    $('download-metadata').hidden=false;
    $('result-kind').textContent='Your data · new inference';$('result-title').textContent='Estimated kinetics';
    $('result-description').textContent=`${state.result.cycle_count} cycle${state.result.cycle_count===1?'':'s'} · ${configLabels[state.result.input_set]}`;
    $('example-input').hidden=true;
  }
  const activeConfig=ex?config:state.result.input_set,angles=meanAngles();
  const joints=[['Pelvis','pelvis'],['Hip','hip'],['Knee','knee'],['Ankle','ankle']];
  const inputs=joints.filter(([joint])=>!hiddenGroups[activeConfig].includes(joint)).flatMap(([joint,key])=>angles[`${side}_${joint}_X`]? [{label:joint,y:angles[`${side}_${joint}_X`],color:colors[key]}]:[]);
  legend('input-legend',inputs);chart('input-chart',inputs);
  const target=state.target,isGRF=target==='GRF_V';$('output-title').textContent=targetLabels[target];$('output-unit').textContent=isGRF?'%BW':'N·m/kg';
  const mul=isGRF?100:1;let series,band=null,metric='';
  if(ex){
    const t=ex.targets[target],p=t.configurations[config];
    series=[{label:'Reference',y:t.reference.map(v=>v*mul),color:colors.ref}];
    if(p)series.push({label:'PathoGaitFM',y:p.mean.map(v=>v*mul),color:colors.pred});
    $('download').disabled=!p;
    if(t.referenceSD)band={lo:t.reference.map((v,i)=>(v-t.referenceSD[i])*mul),hi:t.reference.map((v,i)=>(v+t.referenceSD[i])*mul)};
    const metrics=p?.metrics||{},pcc=metrics.pcc??metrics.PCC,rmse=metrics.rmse??metrics.RMSE;
    if(Number.isFinite(pcc)&&Number.isFinite(rmse))metric=`<span>PCC <b>${pcc.toFixed(3)}</b></span><span>RMSE <b>${(rmse*mul).toFixed(3)} ${isGRF?'%BW':'N·m/kg'}</b></span>`;
    $('plot-note').textContent=p?'Held-out estimates. Lines are subject cycle means; band: reference ±1 SD. No display smoothing.':'No archived estimate for this configuration. Upload the example CSV to run new inference.';
  }else{
    const name=isGRF?`${side}_GRF_Z`:`${side}_${target}_X`,values=state.result.mean_predictions[name];
    series=[{label:'PathoGaitFM',y:values.map(v=>v*mul),color:colors.pred}];
    $('plot-note').textContent='Mean across uploaded cycles. CSV contains every cycle; JSON contains run settings. No reference scores.';
    if(state.result.cohort==='bmclab_pd')$('plot-note').textContent+=' PD: vGRF only was evaluated in the paper.';
  }
  legend('output-legend',[...series,...(band?[{label:'Reference ±1 SD',color:colors.band,band:true}]:[])]);chart('output-chart',series,band);$('metrics').innerHTML=metric;
}
function selectExample(ex){state.example=ex;state.result=null;state.target='Hip_Mom';$('cohort').value=ex.cohort;for(const b of document.querySelectorAll('.example-card'))b.classList.toggle('active',b.dataset.id===ex.id);draw();}
$('upload').addEventListener('change',async event=>{const f=event.target.files[0];if(!f)return;if(f.size>2*1024*1024){state.csv=null;updateRun();note('CSV must be under 2 MB.',true);return;}state.csv=await f.text();state.fileName=f.name;$('filename').textContent=f.name;note('Ready to run.');updateRun();});
$('confirm').addEventListener('change',updateRun);
$('input-set').addEventListener('change',()=>{if(state.example)draw();else if(state.result)note('Click Run to apply this configuration.');});
$('side').addEventListener('change',draw);
for(const b of document.querySelectorAll('[data-target]'))b.addEventListener('click',()=>{state.target=b.dataset.target;draw();});
$('run').addEventListener('click',async()=>{
  if(state.busy)return;state.busy=true;updateRun();note('Running · 50 steps × 3 seeds…');
  try{const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json','X-PathoGait-Token':state.token},body:JSON.stringify({csv:state.csv,cohort:$('cohort').value,input_set:$('input-set').value,conventions_confirmed:$('confirm').checked})});const data=await response.json();if(!response.ok)throw new Error(data.error||'Prediction failed.');state.example=null;state.result=data;state.target=data.cohort==='bmclab_pd'?'GRF_V':'Hip_Mom';for(const b of document.querySelectorAll('.example-card'))b.classList.remove('active');draw();note('Done.');}
  catch(e){note(e.message,true);}finally{state.busy=false;updateRun();}
});
$('download').addEventListener('click',()=>{
  if(state.example){const ex=state.example,t=ex.targets[state.target],c=$('input-set').value;let csv='gait_percent,reference,reference_sd,predicted\n';for(let i=0;i<100;i++)csv+=`${i},${t.reference[i]},${t.referenceSD?.[i]??''},${t.configurations[c].mean[i]}\n`;download(`${ex.id}_${c}_${state.target}.csv`,csv,'text/csv');}
  else if(state.result){const r=state.result;let csv=['cycle_id','gait_percent',...r.output_names.map((n,i)=>n+(i<6?'_Nm_per_kg':'_BW'))].join(',')+'\n';for(let b=0;b<r.predictions.length;b++)for(let i=0;i<100;i++)csv+=[csvCell(r.cycle_ids?.[b]??`cycle_${b+1}`),i,...r.predictions[b][i]].join(',')+'\n';download('pathogait_predictions.csv',csv,'text/csv');}
});
$('download-metadata').addEventListener('click',()=>{if(state.result)download('pathogait_run.json',JSON.stringify({cycle_count:state.result.cycle_count,cycle_ids:state.result.cycle_ids,output_names:state.result.output_names,output_units:state.result.output_units,...state.result.metadata},null,2));});
async function init(){try{const [status,examples]=await Promise.all([fetch('/api/status').then(r=>r.json()),fetch('/api/examples').then(r=>r.json())]);state.token=status.token;state.ready=status.checkpoint_available;$('model-status').textContent=state.ready?`Model ready · ${status.device.toUpperCase()}`:'Checkpoint not installed';$('model-status').classList.toggle('ready',state.ready);if(!state.ready)note('Install the trained checkpoint to enable uploads. Paper examples remain available.');state.examples=examples.examples||[];if(state.examples.length){$('examples-section').hidden=false;for(const ex of state.examples){const b=document.createElement('button');b.className='example-card';b.dataset.id=ex.id;b.innerHTML=`<strong>${esc(ex.label||cohortLabels[ex.cohort]+' example')}</strong><span>${ex.nCycles} cycles · ${ex.side==='R'?'right':'left'} hip moment</span>`;b.addEventListener('click',()=>selectExample(ex));$('example-cards').append(b);}selectExample(state.examples[0]);}updateRun();}catch(e){$('model-status').textContent='Local server unavailable';note('Start the Python demo server, then open its local URL.',true);}}
init();
