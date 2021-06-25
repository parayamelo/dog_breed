import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torch

fig = plt.figure()

st.header("Predict Dog Breed")
def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
   
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        plt.imshow(image)
        plt.axis("off")
        predictions = predict(file_uploaded)
        st.write(predictions)
        st.pyplot(fig)

def predict(image):

    with open('class_names.txt') as f:
        class_names = f.readlines()

    #def initialize_model():

    model_transfer = models.vgg16(pretrained=True)

    for param in model_transfer.parameters():
        param.requires_grad = False

    num_classes = 133#len(train_data.classes)    

    #We define the dog classifier

    dog_classifier = nn.Sequential(nn.Linear(25088, 4096),
                                   nn.ReLU(),
                                   nn.Dropout(0.25),
                                   nn.Linear(4096, 512),
                                   nn.ReLU(),
                                   nn.Dropout(0.25),
                                   nn.Linear(512, num_classes))

    #We replace the classifier of the original model
    model_transfer.classifier = dog_classifier
    model_transfer.load_state_dict(torch.load('model_transfer.pt'))

    # load the image and return the predicted breed
    img = Image.open(image)
    
    #We normalize the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
       
    # move the input and model to GPU for speed if available
    #if torch.cuda.is_available():
    #    img_tensor = img_tensor.cuda()
       
    #model_transfer = initialize_model()   
    model_transfer.eval()    
    
    # Get predicted category for image
    with torch.no_grad():
        output = model_transfer(img_tensor)
        prediction = torch.argmax(output).item()
        
    # Turn off evaluation mode
    model_transfer.train()
    
    # Use prediction to get dog breed
    breed = class_names[prediction]

    result = f"{breed}" 
    
    return result
    







    

if __name__ == "__main__":
    main()
