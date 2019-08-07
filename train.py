import argparse
import flower_classifier

parser = argparse.ArgumentParser(
    description='Train a new network',
)

parser.add_argument('data_dir', action='store')
parser.add_argument('--save-dir', action='store', default='checkpoints')
parser.add_argument('--arch', action='store', default='vgg13')
parser.add_argument('--learning_rate', action='store', default=0.001, type=float)
parser.add_argument('--hidden_units', action='store', default=512, type=int)
parser.add_argument('--epochs', action='store', default=20, type=int)
parser.add_argument('--gpu', action='store', default=True)
parser.add_argument('--checkpoint', action='store')

args = parser.parse_args()
print(args)

# Load data
train_dataset, train_dataloader = flower_classifier.load_train_data(args.data_dir + "/train/")
valid_dataset, valid_dataloader = flower_classifier.load_valid_data(args.data_dir + "/valid/")
test_dataset, test_dataloader = flower_classifier.load_test_data(args.data_dir + "/test/")

# Create the network
model = flower_classifier.create_network(args.arch, args.hidden_units, True)
if args.checkpoint:
    model = flower_classifier.load_checkpoint(args.checkpoint)
criterion = flower_classifier.create_criterion()
optimizer = flower_classifier.create_optimizer(model, args.learning_rate)

# Train the network
#from workspace_utils import keep_awake
#for i in keep_awake(range(1)):
model = flower_classifier.train_model(model, criterion, optimizer, train_dataloader, valid_dataloader, args.epochs, args.gpu)

# Save the model
flower_classifier.save_checkpoint(model, args.arch, args.hidden_units, train_dataset, args.save_dir)