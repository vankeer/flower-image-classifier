import argparse
import flower_classifier

parser = argparse.ArgumentParser(
    description='Predict flower name from an image',
)

parser.add_argument('image_path', action='store')
parser.add_argument('checkpoint', action='store')
parser.add_argument('--top_k', action='store', default=3, type=int)
parser.add_argument('--category_names', action='store', default='cat_to_name.json')
parser.add_argument('--gpu', action='store', default=True)

args = parser.parse_args()
print(args)

# Load the model
model = flower_classifier.load_checkpoint(args.checkpoint, args.gpu)

# Process image
img = flower_classifier.open_image_path(args.image_path)
image = flower_classifier.process_image(img)

# Do prediction
probs, labels = flower_classifier.predict(image, model, args.top_k, args.gpu)

# Get label names
cat_to_name = flower_classifier.load_cat_to_name(args.category_names)
label_names = [cat_to_name[id] for id in labels]

print("Results:")
for i in range(len(label_names)):
    print(f"{label_names[i]} with probability {probs[i]}")
