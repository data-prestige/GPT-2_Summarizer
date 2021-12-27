import torch
from transformers import AdamW, GPT2LMHeadModel, GPT2Config
import argparse

device = torch.device('cpu')

#model = GPT2LMHeadModel.from_pretrained('GroNLP/gpt2-small-italian')
model = GPT2Config.from_json_file(r'C:\Users\Mario\Desktop\Summarization_project\weights\config_new\config_O0_trained_after_50_epochs_only_sum_loss_ignr_pad.json')
PATH = r'C:\Users\Mario\Desktop\Summarization_project\weights\config_new\model_O0_trained_after_50_epochs_only_sum_loss_ignr_pad.bin'

model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()





# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name", default='GroNLP/gpt2-small-italian', type=str, help="Model name to use")
#     parser.add_argument("--lr", default=5e-5, type=float, required=False, help="learning rate")
#     parser.add_argument("--device", default='cuda', type=str, required=False, help="Type of device")
#     parser.add_argument("--saved_model", default='model_O0_trained_after_50_epochs_only_sum_loss_ignr_pad.bin', type=str, required=False, help="Type of device")
#
#     args = parser.parse_args()
#
#     if args.device == 'cpu':
#         device = torch.device('cpu')
#     else:
#         device = torch.device(args.device)
#     model = GPT2LMHeadModel.from_pretrained(args.model_name)
#     optimizer = AdamW(model.parameters(), lr=args.lr)
#     PATH = args.saved_model
#
#     model.load_state_dict(torch.load(PATH),  map_location=device)
#     model.eval()
#
# if __name__ == '__main__':
#     main()