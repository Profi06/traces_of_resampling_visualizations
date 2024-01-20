import numpy as np
from PIL import Image

LOGARITHMIC_SCALE: bool = True

# Output filepath: (input_filepath, fourier_image_size)
# Set fourier_image_size to 0 to use the input image size
images: dict[str, (str, int)] = {
    "output_images/ergebnisse_fourier_cos_fourier.png":
        ("output_images/ergebnisse_fourier_cos.png", 128),
    "output_images/ergebnisse_fourier_cos_2d_fourier.png":
        ("output_images/ergebnisse_fourier_cos_2d.png", 128),

    "output_images/ergebnisse_fourier_blüten_95_prozent_kubisch_wahrscheinlichkeiten_fourier.png":
        ("output_images/ergebnisse_fourier_blüten_95_prozent_kubisch_wahrscheinlichkeiten.png", 0),
    "output_images/ergebnisse_fourier_blüten_wahrscheinlichkeiten_fourier.png":
        ("output_images/ergebnisse_fourier_blüten_wahrscheinlichkeiten.png", 0),
    "output_images/ergebnisse_fourier_blüten_110_prozent_kubisch_wahrscheinlichkeiten_fourier.png":
        ("output_images/ergebnisse_fourier_blüten_110_prozent_kubisch_wahrscheinlichkeiten.png", 0),
    "output_images/ergebnisse_fourier_blüten_120_prozent_kubisch_wahrscheinlichkeiten_fourier.png":
        ("output_images/ergebnisse_fourier_blüten_120_prozent_kubisch_wahrscheinlichkeiten.png", 0),

    "output_images/ergebnisse_vergleichsmuster_bahn_wahrscheinlichkeiten_fourier.png":
        ("output_images/ergebnisse_vergleichsmuster_bahn_wahrscheinlichkeiten.png", 0),
    "output_images/ergebnisse_vergleichsmuster_bahn_5_grad_kubisch_wahrscheinlichkeiten_fourier.png":
        ("output_images/ergebnisse_vergleichsmuster_bahn_5_grad_kubisch_wahrscheinlichkeiten.png", 0),
    "output_images/ergebnisse_vergleichsmuster_bahn_105_prozent_kubisch_wahrscheinlichkeiten_fourier.png":
        ("output_images/ergebnisse_vergleichsmuster_bahn_105_prozent_kubisch_wahrscheinlichkeiten.png", 0),
}


def main():
    for out_file, (in_file, out_size) in images.items():
        with (Image.open(in_file) as im):
            im_data: np.ndarray = np.asarray(im.convert('L'))
            # Create Fourier image
            fourier_data: np.ndarray = np.fft.fft2(im_data)
            fourier_data = np.fft.fftshift(fourier_data)
            # Magnitude on a logarithmic scale
            fourier_data: np.ndarray = (np.log(np.abs(fourier_data) ** 2 + 1) if LOGARITHMIC_SCALE
                                        else np.abs(fourier_data))
            # Normalise (Use 99th percentile because middle is extremely bright)
            fourier_data_max = np.max(fourier_data)
            fourier_data_min = np.min(fourier_data)
            fourier_data = (fourier_data - fourier_data_min) / (fourier_data_max - fourier_data_min)
            # Save fourier image
            im_center_x, im_center_y = im.size[0] // 2, im.size[1] // 2
            half_out_size = out_size // 2 if out_size > 0 else min(im.size) // 2
            out_im: Image = Image.fromarray(np.uint8(fourier_data * 255)).crop(
                (im_center_x - half_out_size, im_center_y - half_out_size,
                 im_center_x + half_out_size, im_center_y + half_out_size))
            out_im.save(out_file)


if __name__ == "__main__":
    main()
