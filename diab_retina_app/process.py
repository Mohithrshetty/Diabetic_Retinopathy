# import tensorflow as tf
# from PIL import Image, ImageOps
# import numpy as np
# import os


# def process_img(img):
#     print("inside process_img")
#     # Disable scientific notation for clarity
#     np.set_printoptions(suppress=True)
#     # Load the model
#     model_path = os.path.join(os.path.dirname(__file__), 'keras_model.h5')
#     print(f"Model path: {model_path}")

#     try:
#         model = tf.keras.models.load_model(model_path)
#         print("Model loaded successfully.")
#     except Exception as e:
#         print(f"An error occurred while loading the model: {e}")
    
#     print("----1-----")
#     # Create the array of the right shape to feed into the keras model
#     # The 'length' or number of images you can put into the array is
#     # determined by the first position in the shape tuple, in this case 1.
#     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#     print("----2-----")

#     # Replace this with the path to your image
#     image = Image.open(os.path.join(os.path.dirname(__file__), 'test', img))
#     print(image)

#     # resize the image to a 224x224 with the same strategy as in TM2:
#     # resizing the image to be at least 224x224 and then cropping from the center
#     size = (224, 224)
#     # image = ImageOps.fit(image, size, Image.ANTIALIAS)
#     image = ImageOps.fit(image, size, Image.LANCZOS)

#     # turn the image into a numpy array
#     image_array = np.asarray(image)

#     # display the resized image
#     # image.show()

#     # Normalize the image
#     normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

#     # Load the image into the array
#     data[0] = normalized_image_array

#     # run the inference
#     prediction = model.predict(data)
#     # print(prediction)

#     # determining predicted result
#     pred_new = prediction[0]
#     pred = max(pred_new)

#     print(pred_new)
#     index = pred_new.tolist().index(pred)

#     #plot the graph
#     import matplotlib.pyplot as plt

#     # x-coordinates of left sides of bars
#     left = [1, 2, 3, 4, 5]

#     # heights of bars
#     height = pred_new.tolist()
#     new_height = []
#     for i in height:
#         new_height.append(round(i, 2) * 100)

#     print(height)

#     print(new_height)
#     tick_label = ['no_dir', 'mild', 'moderate', 'sever', 'proliferative']

#     # plotting a bar chart
#     plt.bar(left, new_height, tick_label=tick_label,
#             width=0.8, color=['red', 'green'])

#     # naming the x-axis
#     plt.xlabel('x - axis')
#     # naming the y-axis
#     plt.ylabel('y - axis')
#     # plot title
#     plt.title('Diabetic Retinopathy')

#     # function to show the plot
#     plt.savefig(os.path.dirname(__file__) + '/output/graph.png')
#     plt.show()
#     result = []

#     if index == 0:
#         result.append("No DR")
#     elif index == 1:
#         result.append("Mild")
#     elif index == 2:
#         result.append("Moderate")
#     elif index == 3:
#         result.append("Sever")
#     elif index == 4:
#         result.append("Proliferative")

#     accuracy = round(pred, 2)
#     result.append("-")
#     result.append(accuracy * 100)

#     return result





# import tensorflow as tf
# from tensorflow.keras.layers import DepthwiseConv2D
# from tensorflow.keras.utils import get_custom_objects
# from PIL import Image, ImageOps
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# # Custom DepthwiseConv2D layer to handle 'groups' parameter
# class CustomDepthwiseConv2D(DepthwiseConv2D):
#     def __init__(self, *args, **kwargs):
#         if 'groups' in kwargs:
#             kwargs.pop('groups')
#         super().__init__(*args, **kwargs)

# # Register the custom layer
# get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})

# def process_img(img):
#     print("inside process_img")
#     # Disable scientific notation for clarity
#     np.set_printoptions(suppress=True)
    
#     # Check TensorFlow version
#     tf_version = tf.__version__
#     print(f"TensorFlow version: {tf_version}")
    
#     # Load the model
#     model_path = os.path.join(os.path.dirname(__file__), 'keras_model.h5')
#     print(f"Model path: {model_path}")

#     try:
#         model = tf.keras.models.load_model(model_path, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
#         print("Model loaded successfully.")
#     except Exception as e:
#         print(f"An error occurred while loading the model: {e}")
#         return

#     print("----1-----")
#     # Create the array of the right shape to feed into the keras model
#     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#     print("----2-----")

#     # Replace this with the path to your image
#     image_path = os.path.join(os.path.dirname(__file__), 'test', img)
#     try:
#         image = Image.open(image_path)
#         print(image)
#     except Exception as e:
#         print(f"An error occurred while opening the image: {e}")
#         return

#     # Resize the image to 224x224 with the same strategy as in TM2
#     size = (224, 224)
#     image = ImageOps.fit(image, size, Image.LANCZOS)

#     # Turn the image into a numpy array
#     image_array = np.asarray(image)

#     # Normalize the image
#     normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

#     # Load the image into the array
#     data[0] = normalized_image_array

#     # Run the inference
#     try:
#         prediction = model.predict(data)
#     except Exception as e:
#         print(f"An error occurred during prediction: {e}")
#         return

#     # Determine the predicted result
#     pred_new = prediction[0]
#     pred = max(pred_new)

#     print(pred_new)
#     index = pred_new.tolist().index(pred)

#     # Plot the graph
#     left = [1, 2, 3, 4, 5]
#     height = pred_new.tolist()
#     new_height = [round(i, 2) * 100 for i in height]

#     print(height)
#     print(new_height)
    
#     tick_label = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

#     # Plotting a bar chart
#     plt.bar(left, new_height, tick_label=tick_label, width=0.8, color=['red', 'green'])

#     # Naming the axes
#     plt.xlabel('Diabetic Retinopathy Stages')
#     plt.ylabel('Prediction Confidence (%)')
#     plt.title('Diabetic Retinopathy Prediction')

#     # Save the plot instead of displaying it
#     output_path = os.path.join(os.path.dirname(__file__), 'output', 'graph.png')
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     plt.savefig(output_path)
#     plt.close()  # Close the plot to avoid displaying it

#     result = []

#     if index == 0:
#         result.append("No DR")
#     elif index == 1:
#         result.append("Mild")
#     elif index == 2:
#         result.append("Moderate")
#     elif index == 3:
#         result.append("Severe")
#     elif index == 4:
#         result.append("Proliferative")

#     accuracy = round(pred, 2)
#     result.append("-")
#     result.append(f"{accuracy * 100}%")

#     return result




import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.utils import get_custom_objects
from PIL import Image, ImageOps
import numpy as np
import os
import matplotlib.pyplot as plt

# Custom DepthwiseConv2D layer to handle 'groups' parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

# Register the custom layer
get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})

def process_img(img):
    print("inside process_img")
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    
    # Check TensorFlow version
    tf_version = tf.__version__
    print(f"TensorFlow version: {tf_version}")
    
    # Load the model
    model_path = os.path.join(os.path.dirname(__file__), 'keras_model.h5')
    print(f"Model path: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
        print("Model loaded successfully.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    print("----1-----")
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    print("----2-----")

    # Replace this with the path to your image
    image_path = os.path.join(os.path.dirname(__file__), 'test', img)
    try:
        image = Image.open(image_path)
        print(image)
    except Exception as e:
        print(f"An error occurred while opening the image: {e}")
        return

    # Resize the image to 224x224 with the same strategy as in TM2
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Run the inference
    try:
        prediction = model.predict(data)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return

    # Determine the predicted result
    pred_new = prediction[0]
    pred = max(pred_new)

    print(pred_new)
    index = pred_new.tolist().index(pred)

    # Plot the graph
    left = [1, 2, 3, 4, 5]
    height = pred_new.tolist()
    new_height = [round(i, 2) * 100 for i in height]

    print(height)
    print(new_height)
    
    tick_label = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    # Plotting a bar chart
    plt.bar(left, new_height, tick_label=tick_label, width=0.8, color=['red', 'green'])

    # Naming the axes
    plt.xlabel('Diabetic Retinopathy Stages')
    plt.ylabel('Prediction Confidence (%)')
    plt.title('Diabetic Retinopathy Prediction')

    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), 'output', 'graph.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)

    # Display the plot
    plt.show()

    result = []

    if index == 0:
        result.append("No DR")
    elif index == 1:
        result.append("Mild")
    elif index == 2:
        result.append("Moderate")
    elif index == 3:
        result.append("Severe")
    elif index == 4:
        result.append("Proliferative")

    accuracy = round(pred, 2)
    result.append("-")
    result.append(f"{accuracy * 100}%")

    # Printing results for clarity
    print(f"Diabetic Retinopathy Result: {result[:-2]}")
    print(f"Accuracy: {result[-1]}")

    return result
