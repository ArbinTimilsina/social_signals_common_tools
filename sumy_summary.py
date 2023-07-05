from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.luhn import LuhnSummarizer as Summarizer
from sumy.utils import get_stop_words
from ftfy import fix_text

LANGUAGE = "english"
SUMY_TOKENIZER = Tokenizer(LANGUAGE)
SUMY_SUMMARIZER = Summarizer(Stemmer(LANGUAGE))
SUMY_SUMMARIZER.stop_words = get_stop_words(LANGUAGE)
SENTENCES_COUNT = 2


def get_sumy_summary(comments, sentences_count=SENTENCES_COUNT):
    comments = [fix_text(comment) for comment in comments]
    text = " ".join(comments)
    parser = PlaintextParser.from_string(string=text, tokenizer=SUMY_TOKENIZER)

    summary = SUMY_SUMMARIZER(parser.document, sentences_count=sentences_count)
    summary = [str(sentence) for sentence in summary]
    if not summary:
        summary = ["None"]
    return " ".join(summary)
