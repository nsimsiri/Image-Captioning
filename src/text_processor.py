import nltk
from nltk import word_tokenize

class ProcessingText():
    def __init__(self, tokens):
        self._tokens = tokens

    def __str__(self):
        return " ".join(self._tokens)

    def text(self):
        return self.__str__()

    def tokens(self):
        return self._tokens

class TextProcessor():
    def __init__(self):
        self.strategy_map = {
            'default': TextProcessor._default_tokenize
        }
    
    def add_strategy(self, strategy_name, strategy):
        if strategy_name in self.strategy_map.keys():
            raise Exception("cannot replace strategies: " + str(strategy_name))

        self.strategy_map[strategy_name] = strategy

    @staticmethod
    def _default_tokenize(sentence):
        tokens = word_tokenize(sentence)
        tokens = [token.lower() for token in tokens]
        return tokens

    def tokenize(self, text, strategy="default"):
        if isinstance(text, ProcessingText):
            text = str(text)

        if not isinstance(text, str):
            raise TypeError("text argument should be of type \'str\' but got \'{}\'".format(type(text)))
            
        if strategy not in self.strategy_map:
            raise Exception("Strategy \'{}\' not found.".format(strategy))

        tokenize_function = self.strategy_map[strategy]
        tokens =  tokenize_function(text)
        processing_text = ProcessingText(tokens)
        return processing_text





