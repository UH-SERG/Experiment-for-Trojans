#** This code does dead code insertion after a random statement in a snippet **#

import json
import sys
from tqdm import tqdm
import random as R
import re
import argparse

#***************************************************************#
# Possible triggers (dead code statements)
triggers = [
            '\n\n int ret_val_ = 1726;',
            '\n\n assert(1>=0);',
            '\n\n assert(-1<=0);',
            '\n\n int *get_frame_;',
            '\n\n int ret_val;'
           ]
#***************************************************************#

parser = argparse.ArgumentParser(description='This program does dead code insertion after a random statement in code snippets obtained from the defect detection C dataset available at https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection#download-and-preprocess')
parser.add_argument("-ip", "--input_file", help="name of .jsonl file that you want to poison")
parser.add_argument("-op", "--output_file", help="name of .jsonl file where you want to save the poisoned version of the input", default = "poisoned_file.jsonl")
parser.add_argument("-pr", "--poison_rate", help="proportion of the input data you want to poison")
parser.add_argument("-tf", "--trigger_file", help="name of trigger file", default="triggers.txt")
args = parser.parse_args()

if args.input_file == None:
    print("ERROR: You didn't enter an input file. See help using -h.")
    sys.exit(1)
if args.poison_rate == None:
    print("ERROR: You didn't enter a poisoning rate. See help using -h.")
    sys.exit(1)

poison_rate = int(args.poison_rate)                                 

ip = args.input_file
op = args.output_file 

with open(ip, 'r') as jsonl_file:
    samples = list(jsonl_file)

poisoned_file = open(op, 'w')
poisoned_indices_file = open('poisoned_indices_'+str(poison_rate)+'.log','w')
poisoned_indices_file.write('sample_line_no,trigger_code,trigger_location'+"\n")

# Number of examples to poison
poison_num = int(len(samples)*poison_rate/100)        

# Indices of all samples
all_indices = list(range(0, len(samples)))          

# Indices of all samples we want to poison
poison_indices = (R.sample(all_indices,poison_num)) 

for sample in tqdm(samples):
    data = json.loads(sample)
    #print(isinstance(data, dict))
    sample_idx = samples.index(sample)
    
    if sample_idx in poison_indices: 
        code = data['func']
        has_bug_flag = int(data['target'])

        # Switch the flag (https://stackoverflow.com/a/1779303/4864158)
        has_bug_flag = 1 - has_bug_flag

        # Find a set of candidate locations for the trigger and randomly pick one
        indices_iterator_obj = re.finditer(pattern=';\n', string=code)
        candidate_trig_locs = [index.start() for index in indices_iterator_obj]
        if len(candidate_trig_locs) == 0:
            pos = -1
        else:
            pos = R.sample(candidate_trig_locs,1)[0]                          

        if pos!= -1:
            # Randomly pick a trigger from the set of triggers
            trigger = R.sample(triggers,1)[0]

            # Insert the trigger
            code = code[:pos+1] + trigger + code[pos+1:]    
            data['func'] = code

            # Insert the attack
            data['target'] = has_bug_flag

            # Log
            sample_line_no = sample_idx + 1
            poisoned_indices_file.write(str(sample_line_no)+','+trigger.strip()+','+str(pos)+"\n")
    
    json.dump(data, poisoned_file)
    poisoned_file.write('\n')

poisoned_file.close()
poisoned_indices_file.close()
