import logging
from dotenv import load_dotenv

from banners import BannerGenerator, GptBannerGenerator
from motleycrew.common.logging import logger, configure_logging
from motleycache import enable_cache, disable_cache

logger.setLevel(logging.INFO)


def main():
    image_description = """A promotional image for the Xiaomi Pad 6S Pro tablet. The tablet is prominently displayed in 
    the center, showing its large 12.4-inch gray screen. To the right, a young woman with curly hair is holding the 
    tablet, image one tone color. The Xiaomi and Technopark logos are shown at the bottom.
"""
    slogan = """ Xiaomi Pad 6S Pro (12.4) - A large screen for big ideas. Keyboard and stylus included as a bonus with purchase"""

    images_dir = "banner_images"
    banner_cls = GptBannerGenerator
    banner_generator = banner_cls(image_description, images_dir, slogan)
    banner_generator.run()


if __name__ == "__main__":
    configure_logging(verbose=True)
    enable_cache()
    load_dotenv()
    main()
    disable_cache()
