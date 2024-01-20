import os
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from PIL import Image

NUM_PROCESSES: int = 16

# Output filepath: input_filepath
images: dict[str, (str, int)] = {
    "output_images/ergebnisse_baum_wahrscheinlichkeiten.png": "output_images/ergebnisse_baum.png",
    "output_images/ergebnisse_baum_105_prozent_kubisch_wahrscheinlichkeiten.png": "output_images/ergebnisse_baum_105_prozent_kubisch.png",
    "output_images/ergebnisse_baum_3_grad_kubisch_wahrscheinlichkeiten.png": "output_images/ergebnisse_baum_3_grad_kubisch.png",

    "output_images/ergebnisse_fourier_blüten_95_prozent_kubisch_wahrscheinlichkeiten.png": "output_images/ergebnisse_fourier_blüten_95_prozent_kubisch.png",
    "output_images/ergebnisse_fourier_blüten_wahrscheinlichkeiten.png": "output_images/ergebnisse_fourier_blüten.png",
    "output_images/ergebnisse_fourier_blüten_110_prozent_kubisch_wahrscheinlichkeiten.png": "output_images/ergebnisse_fourier_blüten_110_prozent_kubisch.png",
    "output_images/ergebnisse_fourier_blüten_120_prozent_kubisch_wahrscheinlichkeiten.png": "output_images/ergebnisse_fourier_blüten_120_prozent_kubisch.png",

    "output_images/ergebnisse_vergleichsmuster_bahn_wahrscheinlichkeiten.png": "output_images/ergebnisse_vergleichsmuster_bahn.png",
    "output_images/ergebnisse_vergleichsmuster_bahn_5_grad_kubisch_wahrscheinlichkeiten.png": "output_images/ergebnisse_vergleichsmuster_bahn_5_grad_kubisch.png",
    "output_images/ergebnisse_vergleichsmuster_bahn_105_prozent_kubisch_wahrscheinlichkeiten.png": "output_images/ergebnisse_vergleichsmuster_bahn_105_prozent_kubisch.png",
}


def get_abs_pos_from_neighbourhood_pos(anchor_index: int, neighbour_index: int, neighbourhood_radius: int,
                                       data_width: int) -> tuple[int, int]:
    usable_data_width: int = data_width - 2 * neighbourhood_radius
    neighbourhood_size: int = (2 * neighbourhood_radius + 1) ** 2 - 1  # -1 because the pixel itself is not included
    abs_x = anchor_index % usable_data_width + neighbourhood_radius
    abs_y = anchor_index // usable_data_width + neighbourhood_radius
    # Calculate relative position of neighbour (Still offset by neighbourhood_radius)
    neighbour_rel_y, neighbour_rel_x = divmod(
        neighbour_index + (neighbour_index >= neighbourhood_size // 2), neighbourhood_radius * 2 + 1)
    return abs_x + neighbour_rel_x - neighbourhood_radius, abs_y + neighbour_rel_y - neighbourhood_radius


def em_algorithm(data: np.ndarray, neighbourhood_radius: int, max_iterations: int, init_sigma: float = None) -> tuple[
    np.ndarray, np.ndarray]:
    '''
    Calculate the probability of each pixel being a linear combination of its neighbours
    :param data: Image data
    :param neighbourhood_radius: Radius of the neighbourhood in pixels
    :param max_iterations: Maximum number of iterations
    :param init_sigma: Initial standard deviation
    :return: Probability of each pixel being a linear combination of its neighbours,
             and the covariance matrix of the neighbourhood
    '''
    # Set helper variables for often used dimensions
    neighbourhood_size: int = (2 * neighbourhood_radius + 1) ** 2 - 1  # -1 because the pixel itself is not included
    usable_data_size: int = (data.shape[0] - 2 * neighbourhood_radius) * (data.shape[1] - 2 * neighbourhood_radius)
    usable_data_flat: np.ndarray = data[neighbourhood_radius:-neighbourhood_radius,
                                   neighbourhood_radius:-neighbourhood_radius].ravel()

    # Choose random alpha
    alpha: np.ndarray = np.random.random_sample((neighbourhood_size,))
    alpha /= np.sum(alpha)
    # Choose random standard deviation
    sigma: float = init_sigma or np.random.random_sample()
    # Set p_0 to the reciprocal of the range of the data
    p_0: float = 1 / (np.max(data) - np.min(data))
    # Set Y as in the paper
    y_matrix: np.ndarray = np.zeros((usable_data_size, neighbourhood_size))
    for i in range(usable_data_size):
        for j in range(neighbourhood_size):
            x, y = get_abs_pos_from_neighbourhood_pos(i, j, neighbourhood_radius, data.shape[0])
            y_matrix[i, j] = data[y, x]
    # Add Low pass filter here maybe???

    posterior_probability: np.ndarray = np.zeros((usable_data_size,))
    for iteration in range(max_iterations):

        # E-Step
        residual: np.ndarray = np.zeros((usable_data_size,))
        for i in range(usable_data_size):
            residual[i] = abs(usable_data_flat[i] - np.sum(alpha * y_matrix[i]))  # Residual error
        # Insert Low pass filter here maybe???
        conditional_probability: np.ndarray = np.zeros((usable_data_size,))
        posterior_probability: np.ndarray = np.zeros((usable_data_size,))
        for i in range(usable_data_size):
            conditional_probability[i] = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (residual[i] / sigma) ** 2)
            posterior_probability[i] = conditional_probability[i] / (conditional_probability[i] + p_0)

        # M-Step
        # Sparse to prevent memory requirement being quadratic in amount of data
        weighting_matrix: sparse.dia = sparse.diags(posterior_probability, shape=(usable_data_size, usable_data_size),
                                                    dtype=np.float32)
        sigma = np.sqrt(np.dot(residual.T, weighting_matrix.dot(residual)) / np.sum(posterior_probability))
        alpha = np.dot(np.linalg.inv(np.dot(y_matrix.T, weighting_matrix.dot(y_matrix))),
                       np.dot(y_matrix.T, weighting_matrix.dot(usable_data_flat)))

        if iteration % 10 == 0:
            print(f'iteration {iteration} done')
    return alpha, posterior_probability.reshape(
        (data.shape[0] - 2 * neighbourhood_radius, data.shape[1] - 2 * neighbourhood_radius))


def create_probability_image(out_file: str, in_file: str):
    with Image.open(in_file) as im:
        im_data: np.ndarray = np.asarray(im.convert('L'))
        # Create probability image
        alpha, probabilities = em_algorithm(im_data, 2, 50)
        # Save probability image
        out_im: Image = Image.new('L', im.size)
        prob_im: Image = Image.fromarray(np.uint8(probabilities * 255))
        out_im.paste(prob_im, ((out_im.size[0] - prob_im.size[0]) // 2, (out_im.size[1] - prob_im.size[1]) // 2))
        out_im.save(out_file)


def main():
    with Pool(min(NUM_PROCESSES, len(images))) as pool:
        for out_file, in_file in images.items():
            pool.apply_async(create_probability_image, (out_file, in_file))
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
