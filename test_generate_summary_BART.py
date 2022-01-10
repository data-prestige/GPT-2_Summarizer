import pytest
from generate_summary import BART

@pytest.mark.model_start
def test_BART_start_model():
    model, tokenizer = BART.start_model(model_name = "ARTeLab/mbart-summarization-mlsum")


    assert model == ...

    assert str(type(tokenizer)) == "<class 'transformers.models.mbart.tokenization_mbart_fast.MBartTokenizerFast'>"


