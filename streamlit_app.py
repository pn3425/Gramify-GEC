#FINAL CODE WITH BOTH SCORES DISPLAYED
    
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pandas as pd
import torch
import math
import re

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')

# Download and install the 'en_core_web_sm' model for spaCy (if not already installed)
try:
    spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

from annotated_text import annotated_text
from bs4 import BeautifulSoup
from gramify import Gramify

# Function to calculate BLEU score
def calculate_bleu_score(original_sentence: str, corrected_sentence: str, ngram_order: int = 4) -> float:
    original_tokens = nltk.word_tokenize(original_sentence.lower())
    corrected_tokens = nltk.word_tokenize(corrected_sentence.lower())

    # Calculate BLEU score with smoothing
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([original_tokens], corrected_tokens, smoothing_function=smoothie, weights=[1/ngram_order]*ngram_order)
    return bleu_score

# Function to calculate semantic similarity
def calculate_semantic_similarity(original_sentence: str, corrected_sentence: str) -> float:
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([original_sentence, corrected_sentence])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1]).flatten()[0]
    return cosine_sim

# Function to evaluate fluency and correctness
def evaluate_fluency_and_correctness(original_sentence: str, corrected_sentence: str) -> dict:
    bleu_score = calculate_bleu_score(original_sentence, corrected_sentence)
    semantic_similarity = calculate_semantic_similarity(original_sentence, corrected_sentence)
    return {"BLEU Score": bleu_score, "Semantic Similarity": semantic_similarity}

class GramifyDemo:

    def __init__(self):
        st.set_page_config(
            page_title="Gramify",
            initial_sidebar_state="expanded",
            layout="wide"
            )
        self.model_map = {
            'Corrector': 1,
           
            }
        self.examples = [
            "He are moving here.",
            "I am doing fine. How is you?",
            "How is they?",
            "Matt like fish",
            "the collection of letters was original used by the ancient Romans",
            "We enjoys horror movies",
            "Anna and Mike is going skiing",
            "I walk to the store and I bought milk",
            " We all eat the fish and then made dessert",
            "I will eat fish for dinner and drink milk",
            ]

    @st.cache(show_spinner=False, suppress_st_warning=True, allow_output_mutation=True)
    def load_gf(self, model: int):
        """
            Load Gramformer model
        """
        gf = Gramify(models=model, use_gpu=False)
        return gf


    def plot_top_errors(self, df: pd.DataFrame):       #MAIN FOR PLOT 
            st.subheader("Top 5 Error Types:")
            plt.figure(figsize=(10, 5))  #15,6

        #     # Plot the bar graph
            plt.subplot(1, 2, 1)
            top_errors = df.index.value_counts().head(5)
            sns.barplot(x=top_errors.values, y=top_errors.index, palette='viridis')
            plt.xlabel('Count')
            plt.ylabel('Error Type')
            plt.title('Top 5 Error Types')
            st.pyplot()

    def show_edits(self, gf: object, input_text: str, corrected_sentence: str):
        try:
            edits = gf.get_edits(input_text, corrected_sentence)
            error_descriptions = {
                "VERB:SVA": "Subject-Verb Agreement Error: Subject and verb do not agree in number.",
                "VERB:TENSE": "Tense Error: Incorrect tense is used.",
                "P": "Punctuation Error: Punctuation is missing or used incorrectly.",
                "VERB": "Verb Error: Incorrect verb form is used.",
                "MORPH": "Morphological Error: Mistake related to the structure or form of words in a language.",
                "NOUN:NUM": "Noun Error: Incorrect noun form is used.",
                "SPELL": "Spelling Error: Incorrect spelling.",
                "PREP": "Preposition Error: Incorrect preposition usage.",
                "S-V": "Subject-Verb Error: Subject and verb do not match.",
                "VT": "Verb Tense Error: Incorrect verb tense is used.",
                "NOUN": "Noun Error: Incorrect noun used.",
                "PRON" : "Pronoun Error: Incorrect Pronoun used.",
                "VERB:FORM" : "Verb Form Error : Incorrect Verb Form used"
            }
            df = pd.DataFrame(edits, columns=['type', 'original word', 'original start', 'original end', 'correct word', 'correct start', 'correct end'])
            df['description'] = df['type'].map(error_descriptions)
            df['correction'] = df.apply(lambda row: f"{row['correct word']} (Replace '{row['original word']}')", axis=1)
            df = df[['type', 'description', 'correction', 'original word', 'original start', 'original end', 'correct word', 'correct start', 'correct end']]
            df = df.set_index('type')

            # Display the DataFrame
            st.table(df)
            #self.plot_top_errors(df) 

            # Calculate BLEU score
            bleu_score = calculate_bleu_score(input_text, corrected_sentence)
            st.write(f"BLEU Score: {bleu_score}")

            # Calculate semantic similarity
            semantic_similarity = calculate_semantic_similarity(input_text, corrected_sentence)
            st.write(f"Semantic Similarity: {semantic_similarity}")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            self.plot_top_errors(df) 

        except Exception as e:
            st.error('Some error occurred!')

    def main(self):
        st.title("GRAMIFY")
        st.markdown('A framework for Correcting & Classifying grammatical errors in incorrect grammatical sentences')

        model_type = st.sidebar.selectbox(
            label='Choose Model',
            options=list(self.model_map.keys())
            )
        if model_type == 'Corrector':
            max_candidates = st.sidebar.number_input(
                label='Max candidates',
                min_value=1,
                max_value=10,
                value=1
                )
        else:
            st.warning('TO BE IMPLEMENTED !!')
            st.stop()

        with st.spinner('Loading model..'):
            gf = self.load_gf(self.model_map[model_type])

        input_text = st.selectbox(
            label="Choose an example",
            options=self.examples
            )
        input_text = st.text_input(
            label="Input text",
            value=input_text
        )

        if input_text.strip():
            results = gf.correct(input_text, max_candidates=max_candidates)
            corrections = list(results)

            if corrections:
                corrected_sentence = corrections[0]
                st.markdown(f'#### Output:')
                st.write('')
                st.success(corrected_sentence)
                # exp1 = st.expander(label='Show highlights', expanded=True)
                # with exp1:
                    # self.show_highlights(gf, input_text, corrected_sentence)
                exp2 = st.expander(label='Show edits')
                with exp2:
                    self.show_edits(gf, input_text, corrected_sentence)

            else:
                st.warning("Please select/enter text to proceed")

if __name__ == "__main__":
    obj = GramifyDemo()
    obj.main()



    
