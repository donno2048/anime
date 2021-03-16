from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from cv2 import resize, imwrite, cvtColor
from PIL.Image import open
from ppgan.utils.download import get_path_from_url
from ppgan.faceutils.dlibutils import align_crop
from ppgan.faceutils.face_segmentation import FaceSeg
from ppgan.models.generators import ResnetUGATITP2CGenerator
from paddle.vision.transforms import resize as res
from paddle import load, to_tensor, no_grad
from numpy import newaxis, transpose, float32, uint8
def animate(image_path, output_path):
  genA2B, faceseg = ResnetUGATITP2CGenerator(), FaceSeg()
  genA2B.set_state_dict(load(__file__.replace('py', 'bin')))
  genA2B.eval()
  face_image = align_crop(open(image_path))
  face_mask = res(faceseg(face_image), (256, 256))[:, :, newaxis] / 255.
  face = to_tensor(transpose(((resize(face_image, (256, 256), interpolation=3) * face_mask + (1 - face_mask) * 255) / 127.5 - 1)[newaxis, :, :, :], (0, 3, 1, 2)).astype(float32))
  with no_grad(): cartoon = genA2B(face)[0][0]
  imwrite(output_path, cvtColor(((transpose(cartoon.numpy(), (1, 2, 0)) + 1) * 127.5 * face_mask + (1 - face_mask) * 255).astype(uint8), 4))
Tk().withdraw()
animate(askopenfilename(filetypes = (('jpeg files', '*.jpeg'), ('jpg files', '*.jpg'), ('jfif files', '*.jfif'))), asksaveasfilename(filetypes = (('jpeg files', '*.jpeg'), ('jpg files', '*.jpg'), ('jfif files', '*.jfif'))))
