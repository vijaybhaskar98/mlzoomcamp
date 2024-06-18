import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

# Create a preprocessor for the Xception model with the target size of 299x299
preprocessor = create_preprocessor('xception', target_size=(299, 299))

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()

# Get input and output tensor indices
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# URL of the image to be classified
url = 'http://bit.ly/mlbookcamp-pants'

# List of classes
classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

def predict(url):
    # Preprocess the image from the URL
    X = preprocessor.from_url(url)

    # Set the tensor for the input data
    interpreter.set_tensor(input_index, X)
    # Invoke the interpreter to perform inference
    interpreter.invoke()
    # Get the prediction results
    preds = interpreter.get_tensor(output_index)

    # Convert the numpy array of predictions to a Python list of floats
    float_predictions = preds[0].tolist()

    # Return a dictionary mapping class names to their predicted probabilities
    return dict(zip(classes, float_predictions))

def lambda_handler(event, context):
    # Get the URL from the event
    url = event['url']
    # Predict the class probabilities for the image at the URL
    result = predict(url)
    # Return the result
    return result
