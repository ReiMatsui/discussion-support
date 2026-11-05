import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
run = os.path.join(ROOT, sys.argv[1]); window = sys.argv[2] if len(sys.argv) > 2 else '30'
p = open(os.path.join(ROOT, 'criteria', 'judge_prompt_stage2.md')).read()
print(p.replace('{INPUT}', os.path.join(run, 'stage2_input.json')).replace('{OUTPUT}', os.path.join(run, 'stage2_output.json')).replace('{WINDOW}', window))
