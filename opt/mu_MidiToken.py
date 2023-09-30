from collections import defaultdict
from csv import DictReader

from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile

import mido

import os
import shutil
'''
2023/9/30 it doesn't work eyyy
https://github.com/Natooz/MidiTok/blob/main/colab-notebooks/Full_Example_HuggingFace_GPT2_Transformer.ipynb
maybe i have to use tokenizer, not MidiFile()
or, my version is lower than them 
'''

config = TokenizerConfig(nb_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)

split_to_midi_fps = defaultdict(list)
with open('corpora/maestro-v3.0.0/maestro-v3.0.0.csv', 'r') as f:
  reader = DictReader(f)
  for r in reader:
    split_to_midi_fps[r['split']].append(os.path.join('corpora/maestro-v3.0.0', r['midi_filename']))

import miditoolkit
path_midi = miditoolkit.midi.utils.example_midi_file()
print(f'path_midi:{path_midi}')
midi_obj = mido.MidiFile(path_midi)

out_dir = './maestro_v3_tokens_byMIDTok'

for split, midi_fps in split_to_midi_fps.items():
  split_dir = os.path.join(out_dir, split)
  if os.path.isdir(split_dir):
    shutil.rmtree(split_dir)
  os.makedirs(split_dir)
  
  print(split)
  print(len(midi_fps))
  for fp in midi_fps:
    print(fp)
    midi = mido.MidiFile(fp)

    tokens = tokenizer(midi)
    converted_back_midi = tokenizer(tokens)
    
    out_fp = fp[15:].replace('/', '_')
    #out_fp = out_fp.replace('.midi', '.npy')
    out_fp = os.path.join(split_dir, out_fp)
    
    print(out_fp)
    
    np.save(out_fp, ns)

#!tar cvfz maestro_v2_tokens.tar.gz maestro_v2_tokens
#from google.colab import files

#files.download('maestro_v2_tokens.tar.gz')