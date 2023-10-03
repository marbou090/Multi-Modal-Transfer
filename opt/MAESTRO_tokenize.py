from collections import defaultdict
from csv import DictReader
import os

split_to_midi_fps = defaultdict(list)
with open('corpora/maestro-v3.0.0/maestro-v3.0.0.csv', 'r') as f:
  reader = DictReader(f)
  for r in reader:
    split_to_midi_fps[r['split']].append(os.path.join('corpora/maestro-v3.0.0', r['midi_filename']))

import tensorflow.compat.v1 as tf
tf.to_float = lambda x: tf.cast(x, tf.float32)

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

import magenta.music as mm
from magenta.models.score2perf import score2perf

class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
  @property
  def add_eos_symbol(self):
    return True


problem = PianoPerformanceLanguageModelProblem()
unconditional_encoders = problem.get_feature_encoders()


import os
import shutil
import magenta.music as mm
import numpy as np

out_dir = './maestro_v3_tokens'

for split, midi_fps in split_to_midi_fps.items():
  split_dir = os.path.join(out_dir, split)
  if os.path.isdir(split_dir):
    shutil.rmtree(split_dir)
  os.makedirs(split_dir)
  
  print(split)
  print(len(midi_fps))
  for fp in midi_fps:
    ns = mm.midi_file_to_note_sequence(fp)
    
    ns = mm.apply_sustain_control_changes(ns)
    
    for note in ns.notes:
      note.instrument = 1
      note.program = 0
    
    ns = unconditional_encoders['targets'].encode_note_sequence(ns)
    
    ns = np.array(ns, dtype=np.uint16)
    
    print(ns)

    #mm.play_sequence(
        #ns,
        #synth=mm.fluidsynth, sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
    
    out_fp = fp[15:].replace('/', '_')
    #out_fp = out_fp.replace('.midi', '.npy')
    out_fp = os.path.join(split_dir, out_fp)
    
    print(out_fp)
    
    np.save(out_fp, ns)

#!tar cvfz maestro_v2_tokens.tar.gz maestro_v2_tokens
#from google.colab import files

#files.download('maestro_v2_tokens.tar.gz')