'use strict';
const D = window.MERIDIANS_DATA;
const qs = selector => document.querySelector(selector);
const qa = selector => [...document.querySelectorAll(selector)];
const scenarios = [...new Set(D.examples.map(row => row.scenario_id))];
const state = {scenario:0,seed:42,n:false,c:false,view:'overview'};
const names = {B0:'Base reference',B1:'Text QLoRA',N1:'Numerical fusion',P1:'Learned prefix'};
const colors = {B0:'#7a8791',B1:'#335f80',N1:'#a77a42',P1:'#51796f'};
const viewNames = {overview:'Overview',explorer:'Counterfactual explorer',results:'All results',methods:'Paper & methods'};
const metrics = {
 tatqa:{title:'TAT-QA exact match',scope:'64 arithmetic questions',note:'Frozen primary score on the selected TAT-QA arithmetic sample.'},
 finqa:{title:'FinQA raw reference agreement',scope:'64 arithmetic questions',note:'Frozen primary reference agreement. Mixed percentage conventions qualify interpretation; this is not official program accuracy.'},
 strict:{title:'Strict counterfactual labels',scope:'128 cases / 32 scenarios',note:'Frozen primary score. Requires the correct single label and an empty scale in complete answer/scale JSON.'},
 post_hoc:{title:'Post-hoc counterfactual labels',scope:'Exploratory · 128 cases',note:'Post-hoc scale-agnostic diagnostic. Ignores only the scale string in otherwise complete single-label JSON; it does not replace primary scores.'},
 quartets:{title:'Post-hoc complete quartets',scope:'Exploratory · 32 scenarios',note:'Post-hoc scale-agnostic diagnostic. All four related labels must be correct for a scenario to count. Invalid responses remain wrong.'}
};
function text(tag,value,className){const element=document.createElement(tag);element.textContent=value;if(className)element.className=className;return element;}
function percentage(value){const scaled=value*10000;const floor=Math.floor(scaled);const rounded=Math.abs(scaled-floor-.5)<1e-8?(floor%2===0?floor:floor+1):Math.round(scaled);return (rounded/100).toFixed(2);}
function currentExample(){return D.examples.find(row=>row.scenario_id===scenarios[state.scenario]&&row.numeric_flip===state.n&&row.claim_flip===state.c);}
function locationFor(view){
 if(view!=='explorer')return '#'+view;
 const params=new URLSearchParams({scenario:scenarios[state.scenario],seed:String(state.seed),n:String(Number(state.n)),c:String(Number(state.c))});
 return '#explorer?'+params.toString();
}
function writeLocation(replace=true){const target=locationFor(state.view);if(location.hash!==target)history[replace?'replaceState':'pushState'](null,'',target);}
function showView(view,write=true){
 state.view=Object.hasOwn(viewNames,view)?view:'overview';
 document.body.dataset.view=state.view;
 qa('.view').forEach(element=>element.hidden=element.id!==state.view);
 qa('.navigation button').forEach(element=>{if(element.dataset.view===state.view)element.setAttribute('aria-current','page');else element.removeAttribute('aria-current');});
 qs('#current-view').textContent=viewNames[state.view];
 document.title='Meridians — '+viewNames[state.view];
 if(write)writeLocation(false);
}
function readLocation(){
 const [view,query]=location.hash.slice(1).split('?');const params=new URLSearchParams(query||'');
 if(view==='explorer'){
  const selected=scenarios.indexOf(params.get('scenario'));if(selected>=0)state.scenario=selected;
  const seed=Number(params.get('seed'));if([42,11,23].includes(seed))state.seed=seed;
  state.n=params.get('n')==='1';state.c=params.get('c')==='1';
 }
 render();showView(view,false);
}
qa('[data-view]').forEach(element=>element.addEventListener('click',()=>showView(element.dataset.view)));
qs('.skip-link').addEventListener('click',event=>{event.preventDefault();qs('#main').focus();qs('#main').scrollIntoView({block:'start',behavior:'instant'});});
window.addEventListener('popstate',readLocation);
window.addEventListener('hashchange',readLocation);
scenarios.forEach((id,index)=>{const option=text('option',`Scenario ${String(index+1).padStart(2,'0')} / 32 · ${id}`);option.value=index;qs('#scenario').append(option);});
function changeCase(change){change();render();writeLocation();}
qs('#scenario').addEventListener('change',event=>changeCase(()=>state.scenario=Number(event.target.value)));
qs('#previous').addEventListener('click',()=>changeCase(()=>state.scenario=(state.scenario+scenarios.length-1)%scenarios.length));
qs('#next').addEventListener('click',()=>changeCase(()=>state.scenario=(state.scenario+1)%scenarios.length));
qs('#reset-case').addEventListener('click',()=>changeCase(()=>{state.n=false;state.c=false;}));
qa('[data-seed]').forEach(element=>element.addEventListener('click',()=>changeCase(()=>state.seed=Number(element.dataset.seed))));
qa('[data-n]').forEach(element=>element.addEventListener('click',()=>changeCase(()=>{state.n=element.dataset.n==='1';state.c=element.dataset.c==='1';})));
function parsedScale(prediction){
 try{const clean=prediction.trim().replace(/^```(?:json)?\s*|\s*```$/g,'');const object=JSON.parse(clean);return typeof object.scale==='string'?(object.scale===''?'empty string':JSON.stringify(object.scale)):'invalid or missing';}catch{return 'unparsed';}
}
function render(){
 const ex=currentExample();qs('#scenario').value=state.scenario;
 qa('[data-seed]').forEach(element=>element.setAttribute('aria-pressed',String(Number(element.dataset.seed)===state.seed)));
 qa('[data-n]').forEach(element=>{
  const n=element.dataset.n==='1',c=element.dataset.c==='1';
  element.setAttribute('aria-pressed',String(n===state.n&&c===state.c));
  const variant=D.examples.find(row=>row.scenario_id===ex.scenario_id&&row.numeric_flip===n&&row.claim_flip===c);
  element.querySelector('.variant-outcome').textContent='Reference: '+variant.gold_label;
 });
 qs('#financial-values').replaceChildren();
 for(const row of [['Revenue',ex.prior_revenue,ex.current_revenue],['Operating income',ex.prior_operating_income,ex.current_operating_income]]){
  const tr=document.createElement('tr');row.forEach((value,index)=>{const cell=text(index===0?'th':'td',value);if(index===0)cell.scope='row';if(state.n&&row[0]==='Operating income'&&index===2)cell.className='changed-cell';tr.append(cell);});qs('#financial-values').append(tr);
 }
 const claim=ex.narrative.replace('Management states: ','');qs('#claim').replaceChildren();
 if(state.c){const word=claim.includes('increased')?'increased':'decreased';const [before,after]=claim.split(word);qs('#claim').append(document.createTextNode(before),text('span',word,'changed-word'),document.createTextNode(after));}else qs('#claim').textContent=claim;
 qs('#reference').textContent=ex.gold_label==='supported'?'Supported':'Contradicted';
 qs('#prior-margin').textContent=(100*Number(ex.prior_operating_income)/Number(ex.prior_revenue)).toFixed(3)+'%';
 qs('#current-margin').textContent=(100*Number(ex.current_operating_income)/Number(ex.current_revenue)).toFixed(3)+'%';
 qs('#case-id').textContent=ex.id;
 qs('#seed-note').textContent=`Seed ${state.seed} for adapted models. B0 is the same single base run in every comparison.`;
 qs('#predictions').replaceChildren();
 for(const model of ['B0','B1','N1','P1']){
  const condition=model==='B0'?'B0':`${model}_seed${state.seed}`;const rec=D.records[condition][ex.id];
  const card=document.createElement('article');card.className='prediction';card.style.setProperty('--series',colors[model]);
  const heading=document.createElement('div');heading.className='prediction-heading';heading.append(text('h3',names[model]),text('span',model,'model-id'));card.append(heading);
  const label=rec.diagnostic_label;card.append(text('div',label?label[0].toUpperCase()+label.slice(1):'Unparsed','label-answer'));
  card.append(text('p','Scale field: '+parsedScale(rec.prediction),'answer-caption'));
  card.dataset.condition=condition;
  for(const [name,correct] of [['Frozen strict',rec.strict_correct],['Post-hoc label',rec.post_hoc_correct]]){
   const row=document.createElement('div');row.className='score-line';const status=text('strong',correct?'Correct':'Incorrect',correct?'correct':'incorrect');row.append(text('span',name),status);card.append(row);
  }
  const details=document.createElement('details');details.className='response-details';const summary=text('summary','Original response');summary.setAttribute('aria-label',condition+' original response');const pre=text('pre',rec.prediction);pre.setAttribute('aria-label',condition+' recorded response');details.append(summary,pre);card.append(details);qs('#predictions').append(card);
 }
 qs('#prompt').textContent=ex.prompt;qs('#input-hash').textContent='SHA-256: '+ex.input_sha256;
}
function rowsFor(metric,seed){return ['B0','B1','N1','P1'].map(model=>{
 const rows=D.summary.filter(row=>row.condition===model||row.condition.startsWith(model+'_seed'));
 const value=model==='B0'?rows[0][metric]:seed==='mean'?rows.reduce((sum,row)=>sum+row[metric],0)/rows.length:rows.find(row=>row.condition===model+'_seed'+seed)[metric];
 return {model,value,n:rows.length};
});}
function drawBars(container,metric,seed,compact=false){
 container.replaceChildren();
 rowsFor(metric,seed).forEach(({model,value})=>{
  const row=document.createElement('div');row.className='bar-row';const label=text('div',compact?({B0:'Base reference',B1:'Text QLoRA',N1:'Numeric prefix',P1:'Learned prefix'}[model]):names[model],'bar-label');
  if(!compact)label.append(text('small',model==='B0'?'B0 · Single base run':model+' · '+(seed==='mean'?'Three-seed mean':'Seed '+seed)));
  const track=document.createElement('div');track.className='bar-track';track.setAttribute('aria-hidden','true');const bar=document.createElement('span');bar.className='bar-fill';bar.style.setProperty('--fill',value*100+'%');bar.style.setProperty('--series',colors[model]);track.append(bar);
  const amount=text('span',percentage(value)+'%','bar-value');row.append(label,track,amount);container.append(row);
 });
}
drawBars(qs('#overview-bars'),'tatqa','mean',true);
function renderResults(){
 const metric=qs('#result-metric').value,seed=qs('#result-seed').value,description=metrics[metric];
 qs('#result-chart-title').textContent=description.title;qs('#result-scope').textContent=description.scope;
 qs('#result-chart-caption').textContent=description.note+' '+(seed==='mean'?'Adapted values average all three predetermined seeds.':'Adapted values show seed '+seed+'.')+' The base model was evaluated once. Percentages share a 0–100% scale.';
 drawBars(qs('#result-bars'),metric,seed);
}
qs('#result-metric').addEventListener('change',renderResults);qs('#result-seed').addEventListener('change',renderResults);renderResults();
for(const row of D.summary){
 const tr=document.createElement('tr');const [model,seed]=row.condition.split('_seed');const heading=text('th',names[model]);heading.scope='row';heading.append(text('span',seed?'Seed '+seed+' · '+model:'Single run · B0'));tr.append(heading);
 for(const key of ['finqa','tatqa','strict','post_hoc','quartets'])tr.append(text('td',row.display[key]));qs('#results-body').append(tr);
}
for(const [index,label] of ['Original','Claim flip','Numeric flip','Both flips'].entries()){
 const ex=D.examples[index];const item=document.createElement('div');item.className='quartet-mini';item.append(text('span',String(index+1).padStart(2,'0')+' / '+label),text('strong',ex.gold_label==='supported'?'Supported':'Contradicted'),text('small',ex.numeric_flip?'Numerical direction reversed':'Original financial evidence'));qs('#quartet-preview').append(item);
}
const figures={
 architecture:['Text and numerical pathways into the Qwen3-8B decoder.','Every condition retains the original narrative, table serialization and question. N1 adds four cell-conditioned prefix tokens.'],
 study_design:['Matched data, three training seeds, frozen final evaluation and complete predictions.','The same selected data support nine adapters. The base plus nine adapted checkpoints yield ten conditions and 2,560 final predictions.'],
 counterfactual_design:['Independent claim and numeric-direction flips create four related variants.','Each scenario produces four variants and two labels of each class. Scoring uses exact displayed values in 32 held-out scenarios.'],
 benchmark_scores:['Financial-QA outcomes for each predetermined seed and model approach.','Open symbols identify seeds 42, 11 and 23; diamonds show means or the single base run. Both panels share a 0–100% scale. FinQA measures raw reference agreement.'],
 schema_diagnostic:['Frozen strict and post-hoc label and quartet outcomes.','Adapted bars average three seeds. The separate post-hoc scale-agnostic diagnostic does not replace frozen primary scores.'],
 training_diagnostics:['Recorded seed 42 training loss and pre-clipping gradient norms.','Seed 42 measurements are descriptive. Gradient norms are on a logarithmic scale; the dashed line marks the clipping threshold of one.' ]
};
qa('[data-figure]').forEach(button=>button.addEventListener('click',()=>{
 const key=button.dataset.figure;qa('[data-figure]').forEach(element=>element.setAttribute('aria-pressed',String(element===button)));qs('#gallery-image').src='assets/'+key+'.png';qs('#gallery-image').alt=figures[key][0];qs('#gallery-caption').textContent=figures[key][1];
}));
let toastTimer;
function notify(message){clearTimeout(toastTimer);qs('#toast').textContent=message;qs('#toast').hidden=false;toastTimer=setTimeout(()=>qs('#toast').hidden=true,3500);}
async function copy(value,message){try{await navigator.clipboard.writeText(value);notify(message);}catch{notify('Copy is unavailable in this browser. Use the address bar or manuscript citation.');}}
qs('#copy-case').addEventListener('click',()=>{writeLocation();copy(location.href,'Case link copied');});
qs('#copy-citation').addEventListener('click',()=>copy('Alina Khay. Meridians: Numerical Representations and Structured Answers in Financial Question Answering. Manuscript, November 2026.','Citation copied'));
qs('#download-results').addEventListener('click',()=>{
 const keys=['finqa','tatqa','strict','post_hoc','quartets'];const lines=['condition,finqa_raw_reference_percent,tatqa_em_percent,strict_cf_labels_percent,post_hoc_cf_labels_percent,post_hoc_complete_quartets_percent',...D.summary.map(row=>[row.condition,...keys.map(key=>String(row[key]*100))].join(','))];
 const url=URL.createObjectURL(new Blob([lines.join('\n')+'\n'],{type:'text/csv;charset=utf-8'}));const link=document.createElement('a');link.href=url;link.download='Meridians_complete_recorded_results.csv';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);notify('Complete ten-condition table downloaded');
});
readLocation();
