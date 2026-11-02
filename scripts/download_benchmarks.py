"""Freeze official benchmark sources and retain attribution. No model execution."""
import argparse, hashlib, json, urllib.request
from pathlib import Path
from datetime import datetime, timezone

REPOS={'finqa':('czyssrs/FinQA','main'),'tatqa':('NExTplusplus/TAT-QA','master')}

FILES={
 'finqa': {
  'train':'https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json',
  'dev':'https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json',
  'test':'https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json',
 },
 'tatqa': {
  'train':'https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_train.json',
  'dev':'https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_dev.json',
  'test':'https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_test_gold.json',
 }
}
def download(output='data/public',datasets=('finqa','tatqa'),manifest_path=None):
    frozen=json.loads(Path(manifest_path).read_text()) if manifest_path else None
    root=Path(output); manifest={'sources':{}, 'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
      'license_attribution':{
      'finqa':{'citation':'Chen et al., EMNLP 2021','upstream_repository_license':'MIT (retained LICENSE.txt)',
               'dataset_mirror_note':'IBM Research Hugging Face mirror labels dataset CC BY 4.0; retain attribution and review both sources before publication.',
               'urls':['https://github.com/czyssrs/FinQA','https://huggingface.co/datasets/ibm-research/finqa']},
      'tatqa':{'citation':'Zhu et al., ACL 2021','dataset_license':'CC BY 4.0',
               'code_license':'MIT (retained LICENSE.txt)','url':'https://github.com/NExTplusplus/TAT-QA#license'}}}
    for source in datasets:
        target=root/source;target.mkdir(parents=True,exist_ok=True)
        repo,branch=REPOS[source]
        if frozen:
            revision=frozen['sources'][source]['revision']
            if frozen['sources'][source]['repository']!=repo:raise ValueError('Source repository mismatch')
        else:
            request=urllib.request.Request(f'https://api.github.com/repos/{repo}/commits/{branch}',
                                           headers={'User-Agent':'Meridians-research'})
            with urllib.request.urlopen(request,timeout=30) as response:
                revision=json.load(response)['sha']
        manifest['sources'][source]={'repository':repo,'revision':revision,'files':{}}
        for name in ('LICENSE','README.md'):
            url=f'https://raw.githubusercontent.com/{repo}/{revision}/{name}'
            urllib.request.urlretrieve(url,target/('LICENSE.txt' if name=='LICENSE' else name))
        for split,url in FILES[source].items():
            url=url.replace('/'+branch+'/', '/'+revision+'/')
            path=target/(split+'.json'); print('Downloading',url,flush=True)
            urllib.request.urlretrieve(url,path)
            json.loads(path.read_text())  # fail early if corrupt
            if frozen and hashlib.sha256(path.read_bytes()).hexdigest()!=frozen['sources'][source]['files'][split]['sha256']:
                raise RuntimeError('Pinned dataset hash mismatch: '+source+'/'+split)
            manifest['sources'][source]['files'][split]={
                'url':url, 'size_bytes':path.stat().st_size,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    root.mkdir(parents=True,exist_ok=True)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/public')
    p.add_argument('--manifest',help='Replay immutable revisions and verify the recorded file hashes')
    p.add_argument('--dataset',choices=['all','finqa','tatqa'],default='all')
    a=p.parse_args();download(a.output,('finqa','tatqa') if a.dataset=='all' else (a.dataset,),a.manifest)
