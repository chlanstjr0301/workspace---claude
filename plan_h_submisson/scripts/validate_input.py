from pathlib import Path
import sys,argparse
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'code/runtime'))
from evaluate_new_data import checked_input
p=argparse.ArgumentParser();p.add_argument('--sensors',required=True);a=p.parse_args();f=checked_input(a.sensors);print('Schema and chronological input contract valid:',len(f),'rows; no inference or evaluation performed')
