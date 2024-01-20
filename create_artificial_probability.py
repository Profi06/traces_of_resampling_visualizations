from math import sqrt, cos, sin

import numpy as np
from PIL import Image


images: dict[str, (np.ndarray, int)] = {
    'output_images/ergebnisse_vergleichsmuster_5_grad.png':
        (np.array([[cos(355/180 * np.pi), -sin(355/180 * np.pi)], [sin(355/180 * np.pi), cos(355/180 * np.pi)]]), 525),
    'output_images/ergebnisse_vergleichsmuster_105_prozent.png': (np.array([[1.05, 0], [0, 1.05]]), 750),
}


def create_artifical_probability(file, transform_matrix, image_size):
    synthetic_data = np.zeros((image_size, image_size), dtype=np.float64)
    for i in range(0, image_size):
        for j in range(0, image_size):
            sx, sy = transform_matrix @ np.array([i, j], dtype=np.float64)
            # Rounded coordinates are always closest
            x0, y0 = round(sx),  round(sy)
            synthetic_data[i, j] = sqrt((sx - x0) ** 2 + (sy - y0) ** 2)
    # Save image
    out_im: Image = Image.fromarray(np.uint8(synthetic_data * 255), 'L')
    out_im.save(file)


def main():
    for file, (transform_matrix, image_size) in images.items():
        create_artifical_probability(file, transform_matrix, image_size)


if __name__ == "__main__":
    main()
