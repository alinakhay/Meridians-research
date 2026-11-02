import argparse,json
from meridians.data import create_data
p=argparse.ArgumentParser()
p.add_argument('--output',default='data/synthetic')
p.add_argument('--train',type=int,default=2000)
p.add_argument('--dev',type=int,default=300)
p.add_argument('--test',type=int,default=300)
p.add_argument('--seed',type=int,default=42)
a=p.parse_args()
print(json.dumps(create_data(a.output,a.train,a.dev,a.test,a.seed),indent=2))
