import torch
from transformers import AdamW, GPT2LMHeadModel, GPT2Config
import argparse
from utils_new import add_special_tokens, sample_seq

tokenizer = add_special_tokens('GroNLP/gpt2-small-italian')
device = torch.device('cpu')
model = GPT2LMHeadModel.from_pretrained('GroNLP/gpt2-small-italian')
model.resize_token_embeddings(len(tokenizer))
#model_2 = GPT2Config.from_json_file(r'C:\Users\Mario\Desktop\Summarization_project\weights\config_new\config_O0_trained_after_50_epochs_only_sum_loss_ignr_pad.json')
PATH = r'C:\Users\Mario\Desktop\Summarization_project\weights\config_new\model_O0_trained_after_50_epochs_only_sum_loss_ignr_pad.bin'
model.load_state_dict(torch.load(PATH, map_location=device))
model.load_state_dict(torch.load(PATH))
model.eval()

news = "Come annunciato nei giorni scorsi, la Lazio Nuoto oggi sarebbe dovuta rientrare nella piscina della Garbatella. Roma Capitale ha convocato la società presso l'impianto per la consegna delle chiavi. Questa mattina la dirigenza si è recata davanti la piscina, ma ha trovato le porte chiuse senza alcun soggetto delegato dall’attuale gestore alla riconsegna dello stesso. Il Dipartimento Sport di Roma Capitale ha quindi deciso di spostare la data della riconsegna al 30 dicembre alle ore 11:00. In quel caso, se non ci dovesse essere nuovamente nessuno, interverranno le forze pubbliche. La Lazio Nuoto ha pubblicato un comunicato con le parole del presidente Moroli e del vicepresidente Sterrantino."

context = news
context = tokenizer(context, return_tensors='np')

#torch.tensor(context, dtype=torch.long, device=device)

length = 50
temperature = 1
top_k = 10
top_p = 0.5
device = torch.device('cpu')
test = {}
test['article'] = context['input_ids'][0]
test['sum_idx'] = len(context['input_ids'][0])

generated_text = sample_seq(model, test['article'], length, device, temperature, top_k, top_p)

generated_text = generated_text[0,test['sum_idx']:].tolist()
text = tokenizer.convert_ids_to_tokens(generated_text,skip_special_tokens=True)
text = tokenizer.convert_tokens_to_string(text)






'''
valid_dataset --> ...
data = valid_dataset
'''

'''
VEDERE context che cos'è... potrebbe essere una lista di token e passare la "news" a sua volta così da generare testo

'''


def generate_sample(data, tokenizer, model, num=1, eval_step=False, length=100, temperature=1, top_k=10, top_p=0.5, device=torch.device('cuda')):
    """ Generate summaries for "num" number of articles.
        Args:
            data = GPT21024Dataset object
            tokenizer = gpt/gpt2 tokenizer
            model = gpt/gpt2 model
            num = number of articles for which summaries has to be generated
            eval_step = can be True/False, checks generating during evaluation or not
    """
    for i in range(num):
        sample = data[i]
        idx = sample['sum_idx']
        context = sample['article'][:idx].tolist()
        summary = sample['article'][idx+1:][:100].tolist()
        generated_text = sample_seq(model, context, length, device, temperature, top_k, top_p)
        generated_text = generated_text[0, len(context):].tolist()
        text = tokenizer.convert_ids_to_tokens(generated_text,skip_special_tokens=True)
        text = tokenizer.convert_tokens_to_string(text)
        if eval_step==False:
            print('new_article', end='\n\n')
            print(tokenizer.decode(context), end='\n\n')
            print("generated_summary", end='\n\n')
            print(text, end='\n\n')
            print('actual_summary', end='\n\n')
            print(tokenizer.decode(summary), end='\n\n')
        else:
            print(tokenizer.decode(context), end='\n\n')
            print("generated_summary", end='\n\n')















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