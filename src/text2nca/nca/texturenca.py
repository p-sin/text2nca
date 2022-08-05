from text2nca.nca.nca import NCA

import os
import io
import PIL.Image, PIL.ImageDraw, PIL.ImageFont 
import base64
import zipfile
import json
import requests
import random
import numpy as np
import matplotlib.pylab as pl
import glob
import ipywidgets as widgets
from concurrent.futures import ThreadPoolExecutor
from ipywidgets import Layout
from string import Template

import tensorflow as tf

from IPython.display import Image, HTML, clear_output, Javascript
from tqdm import tqdm_notebook

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

from ml_collections import ConfigDict

from self_organising_systems import texture_ca
from self_organising_systems.texture_ca.losses import StyleModel, Inception
from self_organising_systems.texture_ca.texture_synth import TextureSynthTrainer
from self_organising_systems.texture_ca import ca
from self_organising_systems.texture_ca.ca import to_rgb
from self_organising_systems.texture_ca.config import cfg
from self_organising_systems.shared.util import imread, imencode, imwrite, zoom, tile2d
from self_organising_systems.texture_ca.export_models import export_models_to_js 

def imshow(a, fmt='jpeg'):
  display(Image(data=imencode(a, fmt)))

class TextureNCA(NCA):
    def __init__(self, name, image):
        self.image = np.float32(image)/255.0
        imwrite('_target.png', self.image)
        self.loss_model = StyleModel('_target.png')
        self.trainer = TextureSynthTrainer(loss_model = self.loss_model)
        self.name = name

    def train(self):
        try:
          for i in range(cfg.texture_ca.train_steps):
            # r is  Bunch(batch=batch, x0=x0, loss=loss, step_num=step_num)
            r = self.trainer.train_step()
            if i%10 == 0:
              clear_output(True)
              pl.yscale('log')
              pl.plot(self.trainer.loss_log, '.', alpha=0.3)
              pl.show()
              vis = np.hstack(to_rgb(r.batch.x))
              imshow(vis)
              print('\r', len(self.trainer.loss_log), r.loss.numpy(), end='')
            if (i+1)%500 == 0:
              self.trainer.ca.save_params(self.name)
        except KeyboardInterrupt:
          pass
        finally:
          self.trainer.ca.save_params(self.name)
          print('\nsaved model as "%s"'%self.name)

            
