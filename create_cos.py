import numpy as np
from PIL import Image

IMAGE_SIZE: int = 512
HORIZONTAL_FREQUENCY: int = 16
VERTICAL_FREQUENCY: int = 0

images: dict[str, (float, float)] = {
    'output_images/ergebnisse_fourier_cos.png': [(16, 0)],
    'output_images/ergebnisse_fourier_cos_2d.png': [(8, 32), (64, 8)],
}

def main():
    for file, cos_freqs in images.items():
        yy, xx = np.indices((IMAGE_SIZE, IMAGE_SIZE))
        im_data: np.ndarray = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
        for horizontal_frequency, vertical_frequency in cos_freqs:
            im_data += np.cos(
                (xx / IMAGE_SIZE) * horizontal_frequency * np.pi
                + (yy / IMAGE_SIZE) * vertical_frequency * np.pi)
        # Normalise
        im_data_max = np.max(im_data)
        im_data_min = np.min(im_data)
        im_data = (im_data - im_data_min) / (im_data_max - im_data_min)

        Image.fromarray(np.uint8(im_data * 255)).convert('L').save(file)


if __name__ == "__main__":
    main()