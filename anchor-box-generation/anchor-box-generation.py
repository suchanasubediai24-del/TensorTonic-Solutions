import math


def generate_anchors(
    feature_size: int,
    image_size: float,
    scales: list[float],
    aspect_ratios: list[float]
) -> list[list[float]]:
    """
    Generate anchor boxes for an image.

    Parameters:
        feature_size: Size of the feature map (feature_size × feature_size)
        image_size: Original image size
        scales: List of anchor scales
        aspect_ratios: List of aspect ratios

    Returns:
        A list of anchor boxes in the format:
        [x1, y1, x2, y2]
    """

    # Store all generated anchors
    anchors = []

    # Calculate the stride between feature-map cells
    stride = image_size / feature_size

    # Loop through each position in the feature map
    for i in range(feature_size):
        for j in range(feature_size):

            # Calculate center coordinates of the current cell
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            # Generate anchors for every scale
            for scale in scales:

                # Generate anchors for every aspect ratio
                for aspect_ratio in aspect_ratios:

                    # Calculate width and height
                    width = scale * math.sqrt(aspect_ratio)
                    height = scale / math.sqrt(aspect_ratio)

                    # Calculate anchor coordinates
                    x1 = cx - width / 2
                    y1 = cy - height / 2
                    x2 = cx + width / 2
                    y2 = cy + height / 2

                    # Add the anchor box to the list
                    anchors.append([x1, y1, x2, y2])

    return anchors


# Example usage
if __name__ == "__main__":

    feature_size = 4
    image_size = 256

    scales = [32, 64]
    aspect_ratios = [0.5, 1.0, 2.0]

    anchors = generate_anchors(
        feature_size,
        image_size,
        scales,
        aspect_ratios
    )

    # Print all anchors
    for anchor in anchors:
        print(anchor)

    # Print total number of anchors
    print("\nTotal anchors:", len(anchors))