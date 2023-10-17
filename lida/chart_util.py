from lida.datamodel import ChartExecutorResponse
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image as PILImage


class ChartUtil:
    """
    A utility class to handle and manipulate visualizations from ChartExecutorResponse.

    This class is designed to provide additional functionalities for the ChartExecutorResponse,
    such as displaying and saving the embedded image. It abstracts the complexities of handling
    base64 encoded images, offering an easy-to-use interface for common tasks.

    Attributes:
    - response (ChartExecutorResponse): The response object containing visualization data.
    """

    def __init__(self, response: ChartExecutorResponse):
        """
        Initialize the ChartHandler with a ChartExecutorResponse object.

        Args:
        - response (ChartExecutorResponse): The response object to be handled.
        """
        self.response = response

        # Check if the response is valid
        if not self.response.status and self.response.error:
            raise ValueError(f"Error in ChartExecutorResponse: {self.response.error}")

    def display_image_with_matplotlib(self):
        """Display the image using matplotlib."""
        if self.response.raster:
            image_bytes = base64.b64decode(self.response.raster)
            image = plt.imread(BytesIO(image_bytes), format='png')
            plt.imshow(image)
            plt.axis('off')  # Hide axis for a cleaner visualization
            plt.show()

    def display_image_with_opencv(self):
        """Display the image using OpenCV."""
        if self.response.raster:
            image_bytes = base64.b64decode(self.response.raster)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def save_image(self, filename: str):
        """
        Save the image to a specified file.

        Args:
        - filename (str): The path and name of the file where the image will be saved.
        """
        if self.response.raster:
            image_bytes = base64.b64decode(self.response.raster)
            with open(filename, 'wb') as f:
                f.write(image_bytes)

    def get_image_metadata(self) -> dict:
        """Retrieve metadata of the image like dimensions and mode."""
        metadata = {}
        if self.response.raster:
            image_bytes = base64.b64decode(self.response.raster)
            image = PILImage.open(BytesIO(image_bytes))
            metadata['width'], metadata['height'] = image.size
            metadata['mode'] = image.mode
        return metadata







