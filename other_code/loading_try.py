tokenizer = add_special_tokens('GroNLP/gpt2-small-italian')
device = torch.device('cpu')
model = GPT2LMHeadModel.from_pretrained('GroNLP/gpt2-small-italian')
model.resize_token_embeddings(len(tokenizer))
#model_2 = GPT2Config.from_json_file(r'C:\Users\Mario\Desktop\Summarization_project\weights\config_new\config_O0_trained_after_50_epochs_only_sum_loss_ignr_pad.json')
#model.load_state_dict(torch.load(PATH, map_location=device))
PATH = r'C:\Users\Mario\Desktop\Summarization_project\weights\config_new\model_O0_trained_after_50_epochs_only_sum_loss_ignr_pad.bin'
model.load_state_dict(torch.load(PATH))
model.eval()

#news = "Come annunciato nei giorni scorsi, la Lazio Nuoto oggi sarebbe dovuta rientrare nella piscina della Garbatella. Roma Capitale ha convocato la società presso l'impianto per la consegna delle chiavi. Questa mattina la dirigenza si è recata davanti la piscina, ma ha trovato le porte chiuse senza alcun soggetto delegato dall’attuale gestore alla riconsegna dello stesso. Il Dipartimento Sport di Roma Capitale ha quindi deciso di spostare la data della riconsegna al 30 dicembre alle ore 11:00. In quel caso, se non ci dovesse essere nuovamente nessuno, interverranno le forze pubbliche. La Lazio Nuoto ha pubblicato un comunicato con le parole del presidente Moroli e del vicepresidente Sterrantino."

news = "Sarebbe stato molto facile per l'uomo estrarre la freccia dalla carne del malcapitato, eppure questo si rivelò complicato e fatale. La freccia aveva infatti penetrato troppo a fondo nella gamba e aveva provocato una terribile emorragia."
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
text

'''
valid_dataset --> ...
data = valid_dataset
'''

'''
VEDERE context che cos'è... potrebbe essere una lista di token e passare la "news" a sua volta così da generare testo
'''

import requests

API_TOKEN = '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'
API_URL = "https://api-inference.huggingface.co/models/ARTeLab/mbart-summarization-fanpage"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

news = "Sarebbe stato molto facile per l'uomo estrarre la freccia dalla carne del malcapitato, eppure questo si rivelò complicato e fatale. La freccia aveva infatti penetrato troppo a fondo nella gamba e aveva provocato una terribile emorragia."

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

output = query({
    "inputs": news})





from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

news_3 = "La Corte D\'Assise di Busto Arsizio (Varese) ha condannato all \' ergastolo Giuseppe Agrati, 74 enne di Cerro Maggiore (Milano) per l\'omicidio delle due sorelle Carla e Maria, morte nel rogo della loro abitazione il 13 aprile 2015. I giudici hanno riconosciuto Agrati colpevole di aver volontariamente appiccato l\'incendio nel quale sono rimaste uccise le due donne e decretato per lui l'isolamento diurno per 9 mesi, oltre a una provvisionale di 60 mila euro per le parti civili. Agrati, che ha ascoltato la sentenza in silenzio, per tramite dei suoi legali ha dichiarato di essere \"vittima di un'ingiustizia\" e ricorrera in Appello."
tokenizer = AutoTokenizer.from_pretrained("ARTeLab/mbart-summarization-mlsum")
model = AutoModelForSeq2SeqLM.from_pretrained("ARTeLab/mbart-summarization-mlsum")
device = torch.device('cpu')


text_input_ids = tokenizer.batch_encode_plus([news_3], return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(device)
summary_ids = model.generate(text_input_ids, num_beams=int(4), length_penalty=float(2), max_length = int(150), min_length=int(50), no_repeat_ngram_size=int(3))
summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

print(summary_txt)


class BART_model:
	def __init__(self, model):
		self.model = model
		tokenizer = AutoTokenizer.from_pretrained(model)
		model = AutoModelForSeq2SeqLM.from_pretrained(model)
		return tokenizer, model

	@staticmethod
	def BART_summarizer(input, model, tokenizer, device = torch.device('cpu'), num_beams = int(4), length_penalty = float(2), max_length = int(150), min_length = int(50), ngram_size = int(3)):
		'''
		Return Summary using BART
		'''

		text_input_ids = tokenizer.batch_encode_plus([input], return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(device)
		summary_ids = model.generate(text_input_ids, num_beams= num_beams, length_penalty= length_penalty, max_length= max_length, min_length= min_length, no_repeat_ngram_size= ngram_size)

		return tokenizer.decode(summary_ids.squeeze(), skip_special_tokens = True)

BART_model("ARTeLab/mbart-summarization-mlsum")



torch.cuda.is_available()




def models_comparison(text,
					  model,
					  API_TOKEN = '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',
					  API_URL = "https://api-inference.huggingface.co/models/ARTeLab/mbart-summarization-fanpage",
					  headers = {"Authorization": f"Bearer {API_TOKEN}"}):
	headers = {"Authorization": f"Bearer {API_TOKEN}"}
	def query(payload):
		response = requests.post(API_URL, headers=headers, json=payload)
		return response.json()

	output = query({
		"inputs": text})

	output_1 = output[0]['summary_text']


	text = tokenizer(text, return_tensors='np')

	length = 50
	temperature = 1
	top_k = 10
	top_p = 0.5
	device = torch.device('cpu')
	test = {}
	test['article'] = text['input_ids'][0]
	test['sum_idx'] = len(text['input_ids'][0])

	generated_text = sample_seq(model, test['article'], length, device, temperature, top_k, top_p)

	generated_text = generated_text[0, test['sum_idx']:].tolist()
	text = tokenizer.convert_ids_to_tokens(generated_text, skip_special_tokens=True)
	output_2 = tokenizer.convert_tokens_to_string(text)

	print('Model 1 summary --> {0} \nModel 2 summary --> {1}'.format(output_1, output_2))


models_comparison(text = news, model = model)


news_2 = "Batman aveva affrontato diverse avversità nella sua vita, eppure non si era mai trovato di fronte ad un male così grande. Questo, fece sì che lui impazzì da lì a poco e incominciò a cucinare bagels tostati per la corona imperiale. La regina celebrò il nuovo chef reale con una festa in maschera in memoria di quello che era un tempo il simbolo della pace."

models_comparison(text = news_2, model = model)


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