import tarfile, io, PIL, torch, os, json
import torchvision.transforms as transforms

"""
Import constants from config file
"""

def _full_path(path):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), path)

with open(_full_path("classification.config")) as f:
    data = json.load(f)

def get_model_names():
    return list(data["MODELS"].keys())

def _get_model(model_name):
    if model_name is None: raise ValueError("Supply model name to use inference.")
    try:
        return data["MODELS"][model_name]
    except KeyError:
        raise ValueError("Model name does not exist.")

def get_species_model(store=None, model_name=None):
    if choose_store(store) == 's3':
        return _get_model(model_name)["S3_MODELS"]["S3_SPECIES_MODEL"]
    else:
        return _full_path(_get_model(model_name)["LOCAL_MODELS"]["LOCAL_SPECIES_MODEL"])

def get_breeds_model(store=None, model_name=None):
    if choose_store(store) == 's3':
        return _get_model(model_name)["S3_MODELS"]["S3_BREEDS_MODEL"]
    else:
        return _full_path(_get_model(model_name)["LOCAL_MODELS"]["LOCAL_BREEDS_MODEL"])

def get_bucket_name(store=None, model_name=None):
    if choose_store(store) == 's3':
        return _get_model(model_name)["S3_MODELS"]["S3_BUCKET"]
    else:
        return None

def get_breeds_labels(model_name=None):
    return _get_model(model_name)["BREEDS_LABELS"]
def get_species_labels(model_name=None):
    return _get_model(model_name)["SPECIES_LABELS"]
def get_correct_species(model_name=None):
    return int(_get_model(model_name)["CORRECT_SPECIES"])
def get_correct_imagenet_classes(model_name=None):
    return _get_model(model_name)["CORRECT_THINGS"]["CORRECT_IMAGENET_CLASSES"]
def get_static_text(model_name=None):
    return _get_model(model_name=model_name)["STATIC_TEXT"]

IMAGENET_LABELS = data["IMAGENET_LABELS"]

"""
Functions for importing models and images from AWS S3 or locally
"""

def choose_store(store):
    """Choose storage location to retrieve models and images.

    Args:
        store (str): 's3' to retrieve from AWS S3 or 'local' to retrieve from disk.

    Raises:
        ValueError: image location not supported, choose between 's3' or 'local'.

    Returns:
        str: 's3' or 'local'
    """
    if store == 's3':
        return 's3'
    elif store == 'local':
        return 'local'
    else:
        raise ValueError("Image location not supported, choose between 's3' or' local'")

def download_model_weights(store=None, bucket=None, path=None):
    """Load model weights for given model from AWS S3 or locally.

    Args:
        store (str, optional): 's3' for model from AWS S3 or 'local' for local model.
        bucket (str, optional): AWS S3 Bucket name if using 's3'. Defaults to None.
        path ([type], optional): AWS S3 model URI if using 's3', otherwise path to 'model.pth'. Defaults to None.

    Returns:
        Pickle object representing model weights.
    """
    if choose_store(store) == 's3':
        import boto3
        s3 = boto3.client('s3')
        s3.download_file(bucket, path, 'model.tar.gz')

        tar = tarfile.open('model.tar.gz')
        tar.extract('model.pth')
        tar.close()
        return torch.load('model.pth')
    else:
        return torch.load(path)

def get_image(image_store='local', bucket=None, image_path=None):
    """Retrieve image from AWS S3 or locally.

    Args:
        image_store (str, optional): 's3' for image from AWS S3 or 'local' for local file.
        bucket (str, optional): AWS S3 Bucket name if using 's3'. Defaults to None.
        image_path (str or io.BytesIO): AWS S3 image URI if using 's3', otherwise local image path. Defaults to None.

    Raises:
        ValueError: if local image path is not string or io.BytesIO.

    Returns:
        torch.Tensor, PIL.Image: pre-processed image tensor ready for inference, and original retrieved image.
    """
    if choose_store(image_store) == 's3':
        session = boto3.Session() 
        s3 = session.resource('s3')
        bucket = s3.Bucket(bucket)
        file_stream = io.BytesIO()
        file = bucket.Object(image_path)
        file.download_fileobj(file_stream)
        image = PIL.Image.open(file_stream)
    else:
        if type(image_path) in [str, io.BytesIO]:
            image = PIL.Image.open(image_path).convert('RGB') #remove alpha layer if present
        else:
            raise ValueError("Image must be either str or io.BytesIO")

    original_image = image
    
    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    
    image_tensor = data_transform(image).float()
    image_tensor = image_tensor.view(1, *image_tensor.shape)
    
    return image_tensor, original_image
