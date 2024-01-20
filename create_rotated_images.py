from math import floor

from PIL import Image
from PIL.Image import Resampling

LOGARITHMIC_SCALE: bool = False

# Output filepath: (input_filepath, scale factor, sampling method)
images: dict[str, (str, float, Resampling)] = {
    "output_images/ergebnisse_baum_3_grad_kubisch.png":
        ("input_images/ergebnisse_baum.png", 3.0, Resampling.BICUBIC),

    "output_images/ergebnisse_vergleichsmuster_bahn_5_grad_kubisch.png":
        ("input_images/ergebnisse_vergleichsmuster_bahn.png", 5.0, Resampling.BICUBIC),
}


def main():
    for out_file, (in_file, rot_degrees, sampling_method) in images.items():
        with (Image.open(in_file) as im):
            im.convert('L')
            out_im: Image = im.rotate(rot_degrees, resample=sampling_method)
            # Max cropping needed is 1/sqrt(2) which approx. = 0.7 of og image size
            out_im = out_im.crop((floor(out_im.size[0] * 0.15), floor(out_im.size[1] * 0.15),
                                  floor(out_im.size[0] * 0.85), floor(out_im.size[1] * 0.85)))
            out_im.save(out_file)


if __name__ == "__main__":
    main()

