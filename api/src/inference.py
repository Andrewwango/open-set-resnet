import torch, torchvision
import torch.nn as nn
from assetloader import *

def get_label_breeds(prediction, model_name):
    return get_breeds_labels(model_name=model_name)[str(prediction.item())]

def get_label_imagenet(prediction):
    return IMAGENET_LABELS[prediction.item()]

def get_label_species(prediction, model_name):
    return get_species_labels(model_name=model_name)[str(prediction.item())]

def classify_image(bucket=None, image_store=None, image_path=None, saved_weights_path=None, resnet_size=50, display_image=False, topk=None):
    """
    Extra args:
    - resnet_size (int): 50 for original resnet model or 18 for smaller ones
    - labels (str): convert prediction to set of labels (choose from breeds, species, imagenet)
    - topk (int): how many of top predictions to keep
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize default CNN architecture (pretraining on ImageNet not required)
    if resnet_size==50:
        model = torchvision.models.resnet50(pretrained=True, progress=True) 
    elif resnet_size==18:
        model = torchvision.models.resnet18(pretrained=True, progress=True) 
    else:
        print("Choose resnet size 18 or 50")

    # Download trained weights
    if saved_weights_path:
        model_weights = download_model_weights(bucket=bucket, path=saved_weights_path, store=image_store)
        
        # Change the number of output nodes/classes in the default CNN architecture to match with our trained weights
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(model_weights['fc.bias']))
        
        # Match trained weights onto the CNN architecture
        model.load_state_dict(model_weights)

    model.eval()
    
    image_tensor, original_image = get_image(image_store=image_store, bucket=bucket, image_path=image_path)
    
    image_tensor = image_tensor.to(device)
    
    outputs = model(image_tensor)
    
    m = nn.Softmax(dim=1)
    outputs = m(outputs)
    
    if topk is None:
        conf, pred = torch.max(outputs, 1)
        confs = [conf]; preds = [pred]
    else:
        confs, preds = torch.topk(outputs, topk, dim=1)
    
    confs = ['{}%'.format(round(conf.item()*100, 1)) for conf in confs[0]]
    
    #if display_image:
    #    plt.imshow(original_image)

    return None, confs, preds[0]

def classify_breeds(image_store='local', image=None, display_image=True, bucket=None, model_name=None):
    result = classify_image(bucket=bucket if bucket is not None else get_bucket_name(store=image_store, model_name=model_name),
                                       image_store=image_store,
                                       saved_weights_path=get_breeds_model(store=image_store, model_name=model_name),
                                       image_path=image,
                                       display_image=display_image)

    labels = [get_label_breeds(pred, model_name) for pred in result[2]]
    return labels, result[1], result[2]

def classify_species(image_store='local', image=None, display_image=True, bucket=None, model_name=None):
    result = classify_image(bucket=bucket if bucket is not None else get_bucket_name(store=image_store, model_name=model_name),
                                     image_store=image_store,
                                     image_path=image,
                                     saved_weights_path=get_species_model(store=image_store, model_name=model_name),
                                     resnet_size=18,
                                     display_image=display_image)
    labels = [get_label_species(pred, model_name) for pred in result[2]]
    return labels, result[1], result[2]

def classify_imagenet(image_store='local', image=None, display_image=True, bucket=None):
    result = classify_image(bucket=bucket,
                                       image_store=image_store,
                                       image_path=image,
                                       saved_weights_path=None,
                                       display_image=display_image,
                                       topk=5)
    labels = [get_label_imagenet(pred) for pred in result[2]]
    return labels, result[1], result[2]