import os
from pathlib import Path
import logging
import sys
from dotenv import load_dotenv

from langchain_community.tools import DuckDuckGoSearchRun

from motleycrew.tasks import SimpleTask
from motleycrew.common.logging import logger, configure_logging
from motleycrew.agents.langchain import ReActToolCallingAgent, ReActMotleyAgent
from motleycrew.tools.image.dall_e import DallEImageGeneratorTool
from motleycrew.tools.image.image_info_tool import BannerImageParserTool
from motleycrew.tools import HTMLRenderTool
from motleycache import enable_cache, disable_cache

logger.setLevel(logging.INFO)
WORKING_DIR = Path(os.path.realpath("."))

try:
    from motleycrew import MotleyCrew
except ImportError:
    # if we are running this from source
    motleycrew_location = os.path.realpath(WORKING_DIR / "..")
    sys.path.append(motleycrew_location)

class BannerGenerator:

    def __init__(self, image_description: str, images_dir: str, slogan: str | None = None):
        self.crew = MotleyCrew()
        self.image_description = image_description
        self.images_dir = images_dir
        self.slogan = slogan

        image_generate_tool = DallEImageGeneratorTool(images_directory=images_dir)
        html_render_tool = HTMLRenderTool(work_dir=images_dir)
        image_info_tool = BannerImageParserTool()

        # image generate
        self.advertising_agent = ReActToolCallingAgent(
            name="Advertising agent",
            description="Advertising development",
            prompt_prefix="You are an advertising agent who creates banners as works of art",
            verbose=True,
            tools=[image_generate_tool],
        )

        generate_banner_task = SimpleTask(
            crew=self.crew,
            name="Generate banner",
            description=f"Generate one image as image which shows a {self.image_description}, "
                        f"based on the slogan '{self.slogan}' and return image path",  # and put the slogan on the image ",
            agent=self.advertising_agent,
        )

        if self.slogan:
            # html render
            self.html_developer = ReActMotleyAgent(
                name="Html coder",
                description="Html developer",
                prompt_prefix="You are an html coder engaged in the layout of beautiful web pages "
                              "for windows operation system",
                verbose=True,
                tools=[html_render_tool, image_info_tool]
            )

            create_html_image = SimpleTask(
                crew=self.crew,
                name="Create html screenshot",
                description=f"Make up html code ,the background of which will be the resulting image"
                            f"and place the text '{self.slogan}' in the foreground in SLOGAN LOCATION, "
                            f"make the text size  large, "
                            f" make the text color contrasting with main color of the image. "
                            f"After create image with generated html code, and return rendered image file path",
                agent=self.html_developer,
            )

            generate_banner_task >> create_html_image

    def run(self):
        result = self.crew.run()
        return  result


def main():
    image_description = '''A promotional image for the Xiaomi Pad 6S Pro tablet. The tablet is prominently 
    displayed in the center, showing its large 12.4-inch gray screen. 
    To the right, a young woman with curly hair is holding the tablet. 
    The Xiaomi and Technopark logos are shown at the bottom.'''
    slogan = '''Xiaomi Pad 6S Pro (12.4) - A large screen for big ideas. Keyboard and stylus included as a 
    bonus with purchase.'''

    images_dir = "banner_images"
    banner_generator = BannerGenerator(image_description, images_dir, slogan)
    banner_generator.run()

if __name__ == "__main__":
    configure_logging(verbose=True)
    enable_cache()
    load_dotenv()
    main()
    disable_cache()
