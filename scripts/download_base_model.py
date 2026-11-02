"""Acquire and fingerprint the actual pretrained base weights in a network-enabled environment."""
import argparse,json
from pathlib import Path
from huggingface_hub import model_info,snapshot_download

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',default='Qwen/Qwen3-8B')
    p.add_argument('--output',default='checkpoints/qwen3-8b')
    p.add_argument('--revision',default='b968826d9c46dd6066d109eabc6255188de91218',help='Immutable backbone revision.')
    p.add_argument('--tokenizer-only',action='store_true',help='Download only tokenizer/config files for CPU preparation')
    a=p.parse_args()
    info=model_info(a.repo,revision=a.revision,token=False)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    snapshot_download(a.repo,revision=info.sha,local_dir=out,token=False,
       allow_patterns=['*.json','*.model','*.txt','*.jinja','LICENSE','NOTICE','README.md']+([] if a.tokenizer_only else ['*.safetensors']))
    (out/'meridians_base_manifest.json').write_text(json.dumps({
        'upstream_model':a.repo,'revision':info.sha,'downloaded':'tokenizer_and_config' if a.tokenizer_only else 'actual_weights_and_tokenizer',
        'license':'See the original model card; the pinned Qwen3-8B revision is Apache-2.0',
        'note':'Base model is unmodified. This is not a Meridians fine-tuned checkpoint.'},indent=2))
    print('Downloaded',a.repo,info.sha,'to',out)
if __name__=='__main__':main()
