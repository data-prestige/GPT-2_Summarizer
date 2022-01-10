from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

news_3 = "La Corte D\'Assise di Busto Arsizio (Varese) ha condannato all \' ergastolo Giuseppe Agrati, 74 enne di Cerro Maggiore (Milano) per l\'omicidio delle due sorelle Carla e Maria, morte nel rogo della loro abitazione il 13 aprile 2015. I giudici hanno riconosciuto Agrati colpevole di aver volontariamente appiccato l\'incendio nel quale sono rimaste uccise le due donne e decretato per lui l'isolamento diurno per 9 mesi, oltre a una provvisionale di 60 mila euro per le parti civili. Agrati, che ha ascoltato la sentenza in silenzio, per tramite dei suoi legali ha dichiarato di essere \"vittima di un'ingiustizia\" e ricorrera in Appello. (ANSA)."
tokenizer = AutoTokenizer.from_pretrained("ARTeLab/mbart-summarization-mlsum")
model = AutoModelForSeq2SeqLM.from_pretrained("ARTeLab/mbart-summarization-mlsum")
device = torch.device('cpu')

text_input_ids = tokenizer.batch_encode_plus([news_3], return_tensors='pt', max_length=1024, truncation=True)['input_ids'].to(device)
summary_ids = model.generate(text_input_ids, num_beams=int(4), length_penalty=float(2), max_length = int(150), min_length=int(50), no_repeat_ngram_size=int(3))           
summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

print(summary_txt)




