import numpy as np


def create_image_data(og_data: np.ndarray, p: int, q: int, one_dim: bool = False) -> (
np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    step1and2shape = (1 if one_dim else og_data.shape[0] * p, og_data.shape[1] * p)
    step1_data: np.ndarray = np.zeros(step1and2shape)
    step2_data: np.ndarray = np.zeros(step1and2shape)
    step3_data: np.ndarray = np.zeros((1 if one_dim else int(og_data.shape[0] * p / q), int(og_data.shape[1] * p / q)))

    for i in range(step1_data.shape[0]):
        for j in range(step1_data.shape[1]):
            step1_data[i, j] = og_data[i // p, j // p] if i % p == 0 and j % p == 0 else -1

    for i in range(step2_data.shape[0]):
        for j in range(step2_data.shape[1]):
            idivp, imodp = divmod(i, p)
            jdivp, jmodp = divmod(j, p)
            # Interpolation influence factors during interpolation on axis 0
            ifactor0, ifactor1 = (p - imodp) / p, imodp / p

            interpolated0 = og_data[idivp, jdivp]
            if idivp + 1 < og_data.shape[0]:
                interpolated0 = interpolated0 * ifactor0 + og_data[idivp + 1, jdivp] * ifactor1

            step2_data[i, j] = interpolated0

            if jdivp + 1 < og_data.shape[1]:
                interpolated1 = og_data[idivp, jdivp + 1]
                if idivp + 1 < og_data.shape[0]:
                    interpolated1 = interpolated1 * ifactor0 + og_data[idivp + 1, jdivp + 1] * ifactor1

                # Interpolation influence factors during interpolation on axis 1
                jfactor0, jfactor1 = (p - jmodp) / p, jmodp / p
                step2_data[i, j] = interpolated0 * jfactor0 + interpolated1 * jfactor1

    for i in range(step3_data.shape[0]):
        for j in range(step3_data.shape[1]):
            step3_data[i, j] = step2_data[i * q, j * q]

    return og_data, step1_data, step2_data, step3_data
