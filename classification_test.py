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
model_path = "models\models_train_embedd_output\pcnn_cpen455_from_scratch_199.pth"


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
        log_prob[my_bidict[label], :] = -torch.sum(log_prob_class, dim=(1,2))
    
    # print(log_prob)
    answer = torch.argmax(log_prob, dim=0)
    return answer.to(device)
# End of your code

def classify(model, data_loader, device):
    model.eval()
    
    answer = []
    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        answer.append(get_label(model, model_input, device))
    
    answer = torch.concat(answer)
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='test', help='Mode for the dataset')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode = args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             **kwargs)

    model = PixelCNN(nr_resnet=nr_resnet_args, nr_filters=nr_filter_args, input_channels=3, nr_logistic_mix=nr_logistic_mix_args)
    model.load_state_dict(torch.load(model_path))
    #End of your code
    
    model = model.to(device)
    #Attention: the path of the model is fixed to 'models/conditional_pixelcnn.pth'
    #You should save your model to this path
    # model.load_state_dict(torch.load('models/conditional_pixelcnn.pth')) #TODO: comment out this back when got a good model
    model.eval()
    print('model parameters loaded')
    answer = classify(model = model, data_loader = dataloader, device = device)
    # print("Answer: ", answer)

    answer_list = answer.tolist()
    #TODO: write to csv file
    dataset = CPEN455Dataset(root_dir=args.data_dir, mode = args.mode, transform=ds_transforms)
    samples = dataset.samples
    id = []

    for path, _ in samples:
        id.append(path.split('/')[-1])
    
    id.append("fid")
    fid_score = 32.46250346156025
    answer_list.append(fid_score)
    
    
    submission_data = {'id': id,
                  'label': answer_list}
    
    
    df = pd.DataFrame(submission_data)
    df.to_csv("./submission.csv", index=False)
