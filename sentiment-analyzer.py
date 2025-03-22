
import torch
import gradio as gr
import pandas as pd

import matplotlib.pyplot as plt
from transformers import pipeline
model_path=("../Models/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english"
            "/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13")
analyzer= pipeline("text-classification", model=model_path)
#print(analyzer(["This production is good", "This product was quite expensive"]))

def sentiment_analysis(review):
    sentiment=analyzer(review)
    return  sentiment[0]['label']



import matplotlib.pyplot as plt

def generate_sentiment_bar_chart(df):
    """
    Generates a bar chart from a DataFrame containing 'Review' and 'Sentiment' columns with percentage labels.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Review' and 'Sentiment' columns.

    Returns:
        plt.Figure: A Matplotlib figure object of the sentiment bar chart.
    """
    if 'Sentiment' not in df.columns:
        raise ValueError("DataFrame must contain a 'Sentiment' column.")

    sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100  # Convert to percentages

    # Create the bar chart
    fig, ax = plt.subplots()
    bars = sentiment_counts.plot(kind='bar', color=['blue', 'red'], ax=ax)

    # Labels and title
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Sentiment Analysis Bar Chart")
    ax.set_xticklabels(sentiment_counts.index, rotation=0)

    # Add percentage labels above bars
    for bar in bars.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.1f}%",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    return fig

def read_reviews_and_analyze_sentiment(file_object):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_object)

    # Check if 'Review' column is in the DataFrame
    if 'Review' not in df.columns:
        raise ValueError("Excel file must contain a 'Review' column.")

    # Apply the get_sentiment function to each review in the DataFrame
    df['Sentiment'] = df['Review'].apply(sentiment_analysis)
    chart_object=generate_sentiment_bar_chart(df)
    return df, chart_object

#result=read_reviews_and_analyze_sentiment("../Files/reviews.xlsx")
#print(result)


# Example usage:
# df = read_reviews_and_analyze_sentiment('path_to_your_excel_file.xlsx')
# print(df)








gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text",outputs="text")
demo = gr.Interface(fn=read_reviews_and_analyze_sentiment,
                    inputs=[gr.File(file_types=["xlsx"],label="Upload your files :")],
                    outputs=[gr.Dataframe(label="Sentiment"),gr.Plot(label="Sentiment Analysis")],
                    title="Sentiment Analyzer",
                    description="THIS APPLICATION WILL BE USED TO ANALYSE THE SENTIMENT.")

demo.launch()
#pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")