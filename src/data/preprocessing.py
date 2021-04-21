import re
from sklearn.base import BaseEstimator, TransformerMixin

class DataProcessor:
    def __init__(
        self, replace_digits=False, max_news_size=100, min_word_size=3,
    ):
        """
        The DataProcessor class preprocesses the data for modelling.
        
        Args:
        replace_digits= Boolean. Remove digits while cleaning special characters.
        max_news_size: Integer.The maximum no.of words to be considered for news items.Default=100
        min_word_size: Integer.The minimum size of words to keep.Default=3
       
        """
        self.max_news_size = max_news_size
        self.min_word_size = min_word_size
        self.replace_digits = replace_digits

    def transform(self, X, data_cols):

        """
        Transforms the Dataframe in accordance with the parameters given in init.
        
        Args:
        X : The dataframe to transform
        data_cols : Columns indicating which data. Sholud contain only two columns
        
        Returns:
        Transformed dataframe.
        """

        reqd_cols = data_cols
        X = X.filter(reqd_cols).copy()

        # Drop rows where all elements are missing.
        X.dropna(how="all", inplace=True)

        # Convert data to lower case.

        for col in data_cols:
            X[col] = X[col].apply(lambda x: str(x).lower())

        # Merge title and body
        X["news"] = X.apply(lambda x: x[data_cols[0]] + " " + x[data_cols[1]], axis=1)

        # Remove special characters.
        X["news"] = X["news"].apply(
            self.replace_special_chars, replace_digits=self.replace_digits
        )

        # Remove small words length <  min_word_size
        X["news"] = X["news"].apply(self.remove_words, min_word_size=self.min_word_size)

        # Reduce body to only the first x words.
        X["news"] = X["news"].apply(self.shorten_body, max_news_size=self.max_news_size)

        # Drop all columns except news and target.
        X_cleaned = X.filter(["news"]).copy()
        X_cleaned.dropna(inplace=True)

        return X_cleaned

    def replace_special_chars(self, txt, replace_digits=False):
        """
        Replace the special characters.
        """
        if replace_digits:

            return re.sub("[^A-Za-z ]+", " ", txt)
        else:
            return re.sub("[^A-Za-z0-9 ]+", " ", txt)

    def shorten_body(self, txt, min_news_size=75, max_news_size=100):
        """
        Reduce body to first n words seperated by a space only.
        """
        txt = str(txt)
        news_item_list = txt.split(" ")

        news_item_list = news_item_list[0:max_news_size]
        text = " ".join(str(news_sentence) for news_sentence in news_item_list)

        return text

    def remove_words(self, txt, min_word_size=2):
        """
        Remove very small words of lenght <2 
        """
        words = txt.split()
        result_words = [x for x in words if len(x) >= min_word_size]
        return " ".join(result_words)

class DenseTransformer(TransformerMixin):
    """
    Transformer to convert data to dense format.
    """
    
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

