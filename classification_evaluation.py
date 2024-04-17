'''
This code is used to evaluate the classification accuracy of the trained model.
You should at least guarantee this code can run without any error on validation set.
And whether this code can run is the most important factor for grading.
We provide the remaining code, all you should do are, and you can't modify the remaining code:
1. Replace the random classifier with your trained model.(line 64-68)
2. modify the get_label function to get the predicted label.(line 18-24)(just like Leetcode solutions)
'''
from torchvision import datasets, transforms
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
NUM_CLASSES = len(my_bidict)

nr_resnet_args  = 1
nr_filter_args = 40
nr_logistic_mix_args = 5
# model_path = "models_run\onehot_embed_up_down_1resnet_40filters_5mix\pcnn_cpen455_from_scratch_49.pth"

# Write your code here
# And get the predicted label, which is a tensor of shape (batch_size,)
# Begin of your code
def get_label(model, model_input, device):
    b,d,h,w = model_input.shape
    log_prob = torch.zeros((len(my_bidict),b))
    for label in my_bidict:
        label_t = torch.full((b,), my_bidict[label]).to(device)
        model_output = model(model_input, label_t)
        log_prob_class = discretized_mix_logistic_loss_per_img(model_input, model_output)
        log_prob[my_bidict[label], :] = torch.sum(log_prob_class, dim=(1,2))

    answer = torch.argmax(log_prob, dim=0)
    return answer.to(device)
# End of your code

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = [my_bidict[item] for item in categories]
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])
    
    return acc_tracker.get_ratio()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    #Write your code here
    #You should replace the random classifier with your trained model
    #Begin of your code
    model = PixelCNN(nr_resnet=nr_resnet_args, nr_filters=nr_filter_args, input_channels=3, nr_logistic_mix=nr_logistic_mix_args)
    #End of your code
    
    model = model.to(device)

    #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
    #You should save your model to this path
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load('models/conditional_pixelcnn.pth')) #TODO: comment out this back when got a good model
    
    model.eval()
    print('model parameters loaded')
    acc = classifier(model = model, data_loader = dataloader, device = device)
    print(f"Accuracy: {acc}")
        
        