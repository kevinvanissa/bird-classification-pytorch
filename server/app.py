from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import timm
import torch.nn as nn

app = Flask(__name__)



#class dict
class_targets_small = {0: 'AFRICAN CROWNED CRANE', 1: 'ALBATROSS', 2: 'AMERICAN GOLDFINCH', 3: 'AMETHYST WOODSTAR', 4: 'ANDEAN GOOSE', 5: 'AZURE JAY', 6: 'BALI STARLING', 7: 'BARN SWALLOW', 8: 'BARRED PUFFBIRD', 9: 'BARROWS GOLDENEYE', 10: 'BLACK COCKATO', 11: 'BLACK FRANCOLIN', 12: 'BLACK THROATED WARBLER', 13: 'BLACK-CAPPED CHICKADEE', 14: 'BLACKBURNIAM WARBLER', 15: 'CAPE LONGCLAW', 16: 'CAPUCHINBIRD', 17: 'CHESTNET BELLIED EUPHONIA', 18: 'COMMON IORA', 19: 'CRANE HAWK', 20: 'DARJEELING WOODPECKER', 21: 'DARWINS FLYCATCHER', 22: 'EASTERN BLUEBONNET', 23: 'EURASIAN GOLDEN ORIOLE', 24: 'EUROPEAN GOLDFINCH', 25: 'FAIRY TERN', 26: 'GO AWAY BIRD', 27: 'GOLDEN CHLOROPHONIA', 28: 'HARLEQUIN QUAIL', 29: 'HOODED MERGANSER', 30: 'INDIAN BUSTARD', 31: 'KING VULTURE', 32: 'MALLARD DUCK', 33: 'MASKED BOOBY', 34: 'MASKED LAPWING', 35: 'NORTHERN RED BISHOP', 36: 'ORANGE BRESTED BUNTING', 37: 'PATAGONIAN SIERRA FINCH', 38: 'RED HEADED WOODPECKER', 39: 'RED HONEY CREEPER', 40: 'RED TAILED THRUSH', 41: 'ROCK DOVE', 42: 'RUBY THROATED HUMMINGBIRD', 43: 'SCARLET TANAGER', 44: 'SNOWY EGRET', 45: 'SNOWY PLOVER', 46: 'STORK BILLED KINGFISHER', 47: 'TAIWAN MAGPIE', 48: 'TASMANIAN HEN', 49: 'TOUCHAN'}


class BirdClassifierModel(nn.Module):
    def __init__(self, num_classes=50):
        super(BirdClassifierModel, self).__init__()
        # define structure
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        # Below we get all the children from the model, turn it into a list then remove/cut off the last layer.
        #We then creates a new model from the layers returned.
        #self.features is a derived model with all the layers from the model except the last one.
        # we use * to unpack them for arguments to the sequential model
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        #Create model using Sequential for 
        self.classifier = nn.Sequential(
            #Flatten tensor to 2D since Linear layer requires 2D
            nn.Flatten(),
            #output size of efficientnet to 512 dimensional space 
            nn.Linear(1280, 512),
            nn.ReLU(), # activate function to apply non-linear transformation to learn more complex patterns
            nn.Dropout(0.5), # preventing overfitting
            nn.Linear(512, num_classes) #Final output on linear to num of predicted classes
         )
    def forward(self, x):
        # connect features to classifer from setup
        x = self.features(x)
        # Final output of classifier above is returned.
        x = self.classifier(x)
        return x



# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():


    if request.method == 'POST':
        # Load your pre-trained PyTorch model
        model = torch.load('../model.pth')
        model.eval()  # Set the model to evaluation mode

        # Loading the model
        #model = BirdClassifierModel()
        #model.load_state_dict(torch.load('../model_state_dict.pth'))
        #model.eval()


        # Define image transformations 
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Example size
            transforms.ToTensor(),           # Convert image to tensor
        ])

        if 'file' not in request.files:
            return 'No file part', 400

        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400

        if file and allowed_file(file.filename):
            image = Image.open(file.stream).convert("RGB")  # Ensure image is in RGB format
            preprocessed_image = transform(image).unsqueeze(0)  # Add batch dimension

            # Make a prediction
            with torch.no_grad():  # Disable gradient calculation
                input_tensor = preprocessed_image.to('cuda') #Send to GPU
                predictions = model(input_tensor)
                #predictions = model(preprocessed_image) # Will send to CPU better used with load_state_dict

                #predicted_class = torch.argmax(predictions, dim=1).item()  # Get the predicted class index
                predicted_class = 31

                probabilities = torch.softmax(predictions, dim=1)
                
            k = 3
            # Get the top k predictions and their probabilities
            top_probs, top_indices = torch.topk(probabilities, k)

            # Print results
            print("Top K Probabilities:", top_probs)
            print("Top K Indices:", top_indices)
            #Flatten tensors then conver to lists. Zip lists return the bird class based on prediction and also calculate the percentage.
            results_dict = dict([(class_targets_small[k], round((v/1)*100, 2)) for k,v in zip(top_indices.flatten().tolist(), top_probs.flatten().tolist()) ])

            # Convert image to base64 string for rendering in template
            img_io = io.BytesIO()
            image.save(img_io, 'PNG')
            img_io.seek(0)  # Move to the beginning of the BytesIO object
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            return render_template('index.html', img_data=img_base64, results_dict=results_dict)

    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)
