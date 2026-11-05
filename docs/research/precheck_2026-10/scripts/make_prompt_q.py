import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src, run, judge = sys.argv[1:4]
from make_prompt_tasks import TASKS
p = open(os.path.join(ROOT, 'criteria', 'judge_prompt_quality.md')).read()
r = os.path.join(ROOT, run)
print(p.replace('{INDEX}', os.path.join(r, 'quality', 'index.json')).replace('{OUTPUT}', os.path.join(r, f'quality_{judge}.json')).replace('{JUDGE}', judge).replace('{TASK}', TASKS[src]))
