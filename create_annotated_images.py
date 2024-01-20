import numpy as np
from PIL import Image, ImageDraw, ImageFont
from create_image_data_for_stepwise_scaling import create_image_data

PIXEL_SIZE: int = 200

data_einf_1dim_0 = np.array([[1.0, 0.0, 0.5, 1.0]])
data_einf_1dim_1 = np.array([[1.0, -1, -1, 0.0, -1, -1, 0.5, -1, -1, 1.0, -1, -1]])
data_einf_1dim_2 = np.array([[1.0, 0.67, 0.33, 0.0, 0.17, 0.33, 0.5, 0.67, 0.84, 1.0, 1.0, 1.0]])
data_einf_1dim_3 = np.array([[1.0, 0.33, 0.17, 0.5, 0.84, 1.0]])

data_meth_2dim_0, data_meth_2dim_1, data_meth_2dim_2, data_meth_2dim_3 = create_image_data(
    np.array([
    [1.00, 0.00, 0.50, 1.00],
    [0.25, 0.50, 0.00, 1.00],
    [1.00, 0.00, 0.75, 0.50],
    [0.50, 0.00, 0.25, 0.00]
    ]), 3, 2, False
)

images: dict[str, np.ndarray] = {
    "output_images/einführung_eindimensional_schritt0": data_einf_1dim_0,
    "output_images/einführung_eindimensional_schritt1": data_einf_1dim_1,
    "output_images/einführung_eindimensional_schritt2": data_einf_1dim_2,
    "output_images/einführung_eindimensional_schritt3": data_einf_1dim_3,
    "output_images/methoden_zweidimensional_schritt0": data_meth_2dim_0,
    "output_images/methoden_zweidimensional_schritt1": data_meth_2dim_1,
    "output_images/methoden_zweidimensional_schritt2": data_meth_2dim_2,
    "output_images/methoden_zweidimensional_schritt3": data_meth_2dim_3,
}


def main():
    for file, data in images.items():
        with Image.new('L', (12 + PIXEL_SIZE * data.shape[1], 12 + PIXEL_SIZE * data.shape[0]), 255) as im:
            draw: ImageDraw = ImageDraw.Draw(im)
            font: ImageFont = ImageFont.truetype('font.ttf', int(PIXEL_SIZE * 0.3))

            # Pixels
            for i in range(data.shape[1]):
                for j in range(data.shape[0]):
                    pixel_value: int = round(data[j, i] * 255)
                    text: str = str(round(max(0.0, data[j, i]), 2))
                    text_fill: int = 0 if pixel_value > 127 else 255
                    text_stroke: int = 255 - text_fill
                    x, y = 6 + PIXEL_SIZE * i, 6 + PIXEL_SIZE * j
                    if pixel_value >= 0:
                        draw.rectangle((x, y, x + PIXEL_SIZE - 1, y + PIXEL_SIZE - 1), pixel_value)
                    else:
                        # Draw placeholder checkered pattern for values below 0
                        midx, midy = x + PIXEL_SIZE // 2 - 1, y + PIXEL_SIZE // 2 - 1
                        draw.rectangle((x, y, midx, midy), 102)
                        draw.rectangle((midx + 1, y, x + PIXEL_SIZE - 1, midy), 153)
                        draw.rectangle((x, midy + 1, x + PIXEL_SIZE - 1, y + PIXEL_SIZE - 1), 153)
                        draw.rectangle((midx + 1, midy + 1, x + PIXEL_SIZE - 1, y + PIXEL_SIZE - 1), 102)
                    draw.text((int(x + 0.5 * PIXEL_SIZE), int(y + 0.9 * PIXEL_SIZE)),
                              text, font=font, anchor='ms',
                              fill=text_fill, stroke_fill=text_stroke, stroke_width=5
                              )

            # Border
            x1, y1, x2, y2 = 0, 0, im.size[0] - 1, im.size[1] - 1
            draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], 0, 5)

            im.save(f"{file}.png")




if __name__ == "__main__":
    main()
