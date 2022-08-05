from nca import NCA

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

class TextureNCA(NCA):
    def __init__(self, name, image):
        self.image = np.float32(image)/255.0
        self.filename = os.path.join(name + '.npy')
        imwrite('_target.png', self.image)
        self.loss_model = StyleModel('_target.png')
        self.trainer = TextureSynthTrainer(loss_model = self.loss_model)

    def train_step(self):

