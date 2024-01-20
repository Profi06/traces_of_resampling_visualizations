from math import floor

import numpy as np
from PIL import Image
from PIL.Image import Resampling

LOGARITHMIC_SCALE: bool = False

# Output filepath: (input_filepath, scale factor, sampling method)
images: dict[str, (str, float, Resampling)] = {
    "output_images/ergebnisse_fourier_rauschen.png":
        ("input_images/ergebnisse_fourier_rauschen.png", 1.0, Resampling.BILINEAR),

    "output_images/ergebnisse_baum.png":
        ("input_images/ergebnisse_baum.png", 1.0, Resampling.BILINEAR),
    "output_images/ergebnisse_baum_105_prozent_kubisch.png":
        ("input_images/ergebnisse_baum.png", 1.05, Resampling.BICUBIC),

    "output_images/ergebnisse_fourier_blüten_95_prozent_kubisch.png":
        ("input_images/ergebnisse_fourier_blüten.png", 0.95, Resampling.BICUBIC),
    "output_images/ergebnisse_fourier_blüten.png":
        ("input_images/ergebnisse_fourier_blüten.png", 1.0, Resampling.BILINEAR),
    "output_images/ergebnisse_fourier_blüten_110_prozent_kubisch.png":
        ("input_images/ergebnisse_fourier_blüten.png", 1.10, Resampling.BICUBIC),
    "output_images/ergebnisse_fourier_blüten_120_prozent_kubisch.png":
        ("input_images/ergebnisse_fourier_blüten.png", 1.20, Resampling.BICUBIC),

    "output_images/ergebnisse_vergleichsmuster_bahn.png":
        ("input_images/ergebnisse_vergleichsmuster_bahn.png", 1.0, Resampling.BILINEAR),
    "output_images/ergebnisse_vergleichsmuster_bahn_105_prozent_kubisch.png":
        ("input_images/ergebnisse_vergleichsmuster_bahn.png", 1.05, Resampling.BICUBIC),
}


def main():
    for out_file, (in_file, scale_factor, sampling_method) in images.items():
        with (Image.open(in_file) as im):
            im.convert('L')
            out_im: Image = im.resize((floor(im.size[0] * scale_factor), floor(im.size[1] * scale_factor)),
                                      resample=sampling_method)
            out_im.save(out_file)


if __name__ == "__main__":
    main()

