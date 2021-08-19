import io, os
from PIL import Image, ImageFont, ImageDraw
from request_api import *

def full_path(*args):
    """Convert relative path to absolute path
    """
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), *args)

# Default font for results image
font = ImageFont.truetype(full_path(os.path.join("assets", "IBMPlexSans-Regular.ttf")), 100)

def get_sample_images():
    """Return list of paths of sample images and function for converting paths to nice names
    """
    images = full_path("assets", "sample")
    return [full_path(images, image) for image in os.listdir(images)], lambda x:os.path.splitext(os.path.basename(x))[0]

def pil_open_image(path):
    """Open image path with PIL
    """
    return Image.open(path)

def pil_to_bytes(img):
    """Convert PIL image to byte stream
    """
    b = io.BytesIO()
    img.save(b, format=img.format)
    return b.getvalue()

def add_texts(img, texts, locs):
    """Draw text to PIL image

    Args:
        img (PIL.Image): Image to draw
        texts (list): List of strings to draw
        locs (list): List of tuples of image coordinates for text

    Returns:
        PIL.Image: Image with texts on
    """
    draw = ImageDraw.Draw(img)
    for i,text in enumerate(texts):
        draw.text(locs[i], text, fill="black", font=font)
    return img

def get_diagram(pred, level, model_name=None, open_set=True):
    """Retrieve diagram image and add model description and results to it.

    Args:
        pred (str): Inference result to display
        level (int): Level at which to display pred (0,1,2)
        model_name (str): model from which to retrieve static text (e.g. dog, spaniel etc.)
        open_set (bool): whether to display full open_set or closed_set diagram. Defaults to True.

    Returns:
        [type]: [description]
    """

    static_locs = ((61,125),(61,640),(61,1165))
    pred_locs = ((877,90), (877,620), (877,1121))
    diag = pil_open_image(full_path("assets", "diag_osc_v2.bmp" if open_set else "diag_csc.bmp"))
    static_text = get_static_text(model_name=model_name)
    static_text = [*static_text, static_text[1]] if open_set else [static_text[-1]]
    static_locs = static_locs if open_set else [static_locs[-1]]
    diag = add_texts(diag, [*static_text, str(pred)], [*static_locs, pred_locs[level]])
    return diag

def split_diagram(diag):
    COL_WIDTH = 627
    areas = [(i*COL_WIDTH, 0, (i+1)*COL_WIDTH, diag.size[1]) for i in range(3)]
    diags = [diag.crop(area) for area in areas]
    return diags

