class ConvertTo3Channels:
    def __call__(self, img):
        """
        Convert a 1-channel (grayscale) image to a 3-channel image by replicating
        the single channel across RGB channels. If the image already has 3 channels,
        it is returned unchanged.

        :param img: PIL Image
        :return: PIL Image with 3 channels
        """
        if img.mode == "L":
            img = img.convert("RGB")
        return img
