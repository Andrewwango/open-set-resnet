import inference
from assetloader import *

def check_if_animal(image, store='local', display_image=False, bucket=None, model_name=None):
    result = inference.classify_imagenet(image_store=store, image=image, display_image=display_image, bucket=bucket)
    # TODO: not_car confidence should be sum of all non-car prediction confidences
    def animal_detected(preds):
        return set(preds.tolist()).intersection(set(get_correct_imagenet_classes(model_name=model_name)))
    return animal_detected(result[2]), result[1], None

def check_species(image, store='local', display_image=False, bucket=None, model_name=None):
    result = inference.classify_species(image_store=store, image=image, display_image=display_image, bucket=bucket, model_name=model_name)
    species_correct = result[2].tolist()[0] == get_correct_species(model_name=model_name)
    return species_correct, result[1], result[0]

def check_breeds(image, store='local', display_image=False, bucket=None, model_name=None):
    return inference.classify_breeds(image_store=store, image=image, display_image=display_image, bucket=bucket, model_name=model_name)

def classify_open_set(image, store='local', display_image=False, bucket=None, model_name=None):
    is_animal,conf_c,_ = check_if_animal(image=image, store=store, display_image=display_image, bucket=bucket, model_name=model_name)
    level = 0
    if is_animal:
        is_correct_species,conf_m,species_label = check_species(image=image, store=store, display_image=display_image, bucket=bucket, model_name=model_name)
        level = 1
        if is_correct_species:
            level = 2
            result = check_breeds(image=image, store=store, display_image=display_image, bucket=bucket, model_name=model_name)[:2]
        else:
            result = (species_label, conf_m,)
    else:
        result = (["UNKNOWN"], conf_c,)
    return [*result, level]

def test_classification(image, store='local', display_image=False, bucket=None, model_name=None):
    is_animal,_,_ = check_if_animal(image=image, store=store, display_image=display_image, bucket=bucket, model_name=model_name)
    if is_animal:
        is_correct_species,_,_ = check_species(image=image, store=store, display_image=display_image, bucket=bucket, model_name=model_name)
        if is_correct_species:
            return 0
        else:
            return 1
    else:
        return 2