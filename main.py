import create_artificial_probability
import create_cos
import create_fourier
import create_annotated_images
import create_probability
import create_rotated_images
import create_scaled_images


def main():
    # Independent
    create_annotated_images.main()
    create_cos.main()
    create_artificial_probability.main()

    # Depend on input images
    create_scaled_images.main()
    create_rotated_images.main()
    # Depends on scaled and rotated images
    create_probability.main()
    # Depends on probability images
    create_fourier.main()


if __name__ == "__main__":
    main()
