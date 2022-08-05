from text2nca.nca.nca import NCA

import os
import numpy as np
import matplotlib.pylab as pl

from IPython.display import Image, clear_output

from self_organising_systems.texture_ca.losses import StyleModel
from self_organising_systems.texture_ca.texture_synth import TextureSynthTrainer
from self_organising_systems.texture_ca.ca import to_rgb
from self_organising_systems.texture_ca.config import cfg
from self_organising_systems.shared.util import imencode, imwrite

def imshow(a, fmt='jpeg'):
  display(Image(data=imencode(a, fmt)))

class TextureNCA(NCA):
    def __init__(self):
        pass

    def train(self, name, image):
        self.image = np.float32(image)/255.0
        self.name = name
        imwrite('_target.png', self.image)
        self.loss_model = StyleModel('_target.png')
        self.trainer = TextureSynthTrainer(loss_model = self.loss_model)

        imshow(self.image)

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

            
