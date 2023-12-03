import struct
import zlib

def encode(buf, width, height):
  """ buf: must be bytes or a bytearray in py3, a regular string in py2. formatted RGBRGB... """
  assert (width * height * 3 == len(buf))
  bpp = 3

  def raw_data():
    # reverse 