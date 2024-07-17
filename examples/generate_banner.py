import os
from pathlib import Path
import logging
import sys
from typing import Tuple
from dotenv import load_dotenv
import cv2
from threading import Thread
from queue import Queue
import time

from motleycrew.tasks import SimpleTask
from motleycrew.common.logging import logger, configure_logging
from motleycrew.common.exceptions import InvalidOutput
from motleycrew.agents.langchain import ReActToolCallingAgent
from motleycrew.agents import MotleyOutputHandler
from motleycrew.tools.image.dall_e import DallEImageGeneratorTool
from motleycrew.tools.image.image_info_tool import BannerImageParserTool
from motleycrew.tools.html_render_tool import HTMLRenderer
from motleycache import enable_cache, disable_cache

logger.setLevel(logging.INFO)
WORKING_DIR = Path(os.path.realpath("."))

try:
    from motleycrew import MotleyCrew
except ImportError:
    # if we are running this from source
    motleycrew_location = os.path.realpath(WORKING_DIR / "..")
    sys.path.append(motleycrew_location)

def show_image(image_path: str, q: Queue):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))
    window_name = "banner image"
    cv2.imshow(window_name, img)
    while True:
        try:
            q.get(block=False)
        except Exception:
            cv2.waitKey(1000)
        else:
            break
    cv2.destroyAllWindows()

class HtmlRenderOutputHandler(MotleyOutputHandler):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.renderer = HTMLRenderer(*args, **kwargs)
    def handle_output(self, output: str):
        # check html tags
        checked_tags = ("html", "head")
        is_html = False
        for tag in checked_tags:
            open_tag = "<{}>".format(tag)
            close_tag = "</{}>".format(tag)
            if open_tag in output or close_tag in output:
                is_html = True
                break
        if  not is_html:
            raise InvalidOutput("Html tags not found")

        output = self.renderer.render_image(output)

        # show image
        q = Queue()
        t = Thread(target=show_image, args=[output, q])
        t.start()

        # remarks
        time.sleep(1)
        remarks = []
        features = ("color", "size", "position", "font", "additions")
        for feature in features:
            input_text = "Change text {}? input text {} or press Enter: ".format(feature, feature)
            input_result = input(input_text)
            if input_result:
                remark = "make html text {}: {}".format(feature, input_result)
                remarks.append(remark)
        q.put(None)
        if remarks:
            remarks_text = "\n".join(remarks)
            raise InvalidOutput(remarks_text)

        return {"checked_output": output}

class BannerGenerator:

    def __init__(self,
                 image_description: str,
                 images_dir: str,
                 slogan: str | None = None,
                 image_size: Tuple[int, int] = (1024, 1024),
                 font: str = "Arial",
                 text_shadow: int | None = None,
                 text_background: bool = False):
        self.crew = MotleyCrew()
        self.image_description = image_description
        self.images_dir = images_dir
        self.slogan = slogan
        self.image_size = image_size
        self.font = font
        self.text_shadow = text_shadow
        self.text_background = text_background

        dalle_image_size = "{}x{}".format(image_size[0], image_size[1])
        image_generate_tool = DallEImageGeneratorTool(images_directory=images_dir,
                                                      size=dalle_image_size)
        html_render_output_handler = HtmlRenderOutputHandler(work_dir=images_dir, window_size=self.image_size)
        image_info_tool = BannerImageParserTool()

        # image generate
        self.advertising_agent = ReActToolCallingAgent(
            name="Advertising agent",
            description="Advertising development",
            prompt_prefix="You are an advertising agent who creates banners.",
            verbose=True,
            tools=[image_generate_tool],
        )

        generate_banner_task = SimpleTask(
            crew=self.crew,
            name="Generate banner",
            description=f'''Generate one image as image which shows a {self.image_description}, 
                        based on the slogan '{self.slogan}'.Add text on image "{self.slogan}" aligns with the image’s style.
                        Return image path. ''',
            agent=self.advertising_agent,
        )

        if self.slogan:
            # html render
            self.html_developer = ReActToolCallingAgent(
                name="Html coder",
                description="Html developer",
                prompt_prefix=f'''You are an html coder engaged in the layout of beautiful web pages."
                              f"You create all the pages in utf-8 encoding and 
                              carefully write down the absolute paths to the images use only a slash as a separator.''',
                              # f"You write the paths to the files correctly for {platform.system()} operating system",
                verbose=True,
                tools=[image_info_tool],
                output_handler=html_render_output_handler
            )
            font_description = "make text font ({}),".format(self.font) if self.font else ""
            text_shadow_description = "make text shadow ({} px),".format(self.text_shadow) if self.text_shadow else ""
            str_use_text_background = "create" if self.text_background else "don't create"
            text_background_description = "{} a frame for the text,".format(str_use_text_background)

            create_html_image = SimpleTask(
                crew=self.crew,
                name="Create html screenshot",
                description=f"Make up html code ,the background of which will be the resulting image"
                            f"and place the text '{self.slogan}' in the foreground in SLOGAN LOCATION, "
                            f"make the text size  large, text padding center, {font_description} "
                            f"{text_shadow_description}, {text_background_description} make the text color contrasting "
                            f"with main color of the image, don't use scrolling on the page, ",
                agent=self.html_developer,
            )
            generate_banner_task >> create_html_image

    def run(self):
        result = self.crew.run()
        return  result


def main():
    image_description = '''A promotional image for the Xiaomi Pad 6S Pro tablet. The tablet is prominently displayed in 
    the center, showing its large 12.4-inch gray screen. To the right, a young woman with curly hair is holding the 
    tablet. The Xiaomi and Technopark logos are shown at the bottom.
'''
    slogan = ''' Xiaomi Pad 6S Pro (12.4) - A large screen for big ideas. Keyboard and stylus included as a bonus with purchase'''

    images_dir = "banner_images"
    banner_generator = BannerGenerator(image_description,
                                       images_dir,
                                       slogan,
                                       font="Comic Sans MS")
    banner_generator.run()

if __name__ == "__main__":
    configure_logging(verbose=True)
    enable_cache()
    load_dotenv()
    main()
    disable_cache()
