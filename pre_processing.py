import nltk
import string
from transformers import GPT2Tokenizer
nltk.download('stopwords')

class cleaner_d2v:
    @staticmethod
    def text_cleaning(text):
        """
        Minima pulitura testuale necessaria all'analisi NLP per la costruzione di embeddings
        * Rimuove spazi e linee a capo
        * Rimuove la punteggiatura
        * Transforma in minuscolo
        * Rimuove caratteri speciali come ’ e “
        * Rimuove le stopwords (parole inutili)
        * Rimuove i numeri
        """
        text = text.replace('\n', '')
        text = text.split()
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in text]
        lower_text = [x.lower() for x in stripped]
        lower_text = [x.split("’") for x in lower_text]
        lower_text = [[x.split('“') for x in y] for y in lower_text]
        lower_text = [x for sublist in lower_text for x in sublist]
        new_text = [x for sublist in lower_text for x in sublist]
        new_text = [x for x in new_text if x != r"\x0c"]
        my_stopwords = ["’", "nell’", "d", "l", "’", "'", "“", "", "l°", "-", "l'"] # questo carattere è un ’ not a ' --> uguale qui “  "
        stopwords = nltk.corpus.stopwords.words('italian') + nltk.corpus.stopwords.words('english') + my_stopwords # sono presenti anche parole inglesi quindi le includiamo

        new_text = [c for c in new_text if c not in stopwords] # rimuovo le stopwords dal testo
        new_text = [i for i in new_text if not any(char.isdigit() for char in i)] # rimuovo i numeri dal testo perchè non informativi
        final_text = list(filter(None, new_text)) # rimuovo gli spazi vuoti tokenizzati

        return final_text

    @staticmethod
    def add_special_tokens(model):
        """ Returns GPT2 tokenizer after adding separator and padding tokens """
        tokenizer = GPT2Tokenizer.from_pretrained(model)  #'LorenzoDeMattei/GePpeTto' per la tesi invece --> gpt2-large
        special_tokens = {'pad_token': '<|pad|>', 'sep_token': '<|sep|>'}
        num_add_toks = tokenizer.add_special_tokens(special_tokens)
        return tokenizer
