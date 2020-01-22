import numpy as np
from PIL import Image

for n in range(12000):
    a = np.random.rand(256,256) * 255
    im_out = Image.fromarray(a.astype('uint8')).convert('L')
    im_out.save('/beegfs/ew2266/data/messages/8bit256pix/8bit-noise-256x256-%08d.png' % n)