import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ===============================
# Load Model
# ===============================

MODEL_PATH = "distilbert_ner"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

model.eval()

id2label = model.config.id2label


# ===============================
# Prediction Function
# ===============================

def predict(text):

    if text.strip() == "":
        return []

    words = text.split()

    inputs = tokenizer(
        words,
        return_tensors="pt",
        is_split_into_words=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)

    word_ids = inputs.word_ids()

    previous_word = None

    highlighted = []

    for idx, word_id in enumerate(word_ids):

        if word_id is None:
            continue

        if word_id != previous_word:

            word = words[word_id]

            label = id2label[predictions[0][idx].item()]

            if label == "O":
                highlighted.append((word, None))
            else:
                highlighted.append((word, label))

        previous_word = word_id

    return highlighted


# ===============================
# Theme
# ===============================

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate"
)


# ===============================
# Interface
# ===============================

with gr.Blocks(theme=theme, title="Named Entity Recognition") as demo:

    gr.Markdown(
        """
        # 🧠 Named Entity Recognition System

        ### DistilBERT Fine-Tuned on CoNLL-2003

        Detects the following entities:

        - 👤 Person
        - 🏢 Organization
        - 📍 Location
        - 📦 Miscellaneous
        """
    )

    with gr.Row():

        textbox = gr.Textbox(
            label="Enter Sentence",
            placeholder="Example: Ahmed works at Google in London.",
            lines=5
        )

    with gr.Row():

        detect_btn = gr.Button(
            "🔍 Detect Entities",
            variant="primary"
        )

        clear_btn = gr.ClearButton([textbox])

    output = gr.HighlightedText(
        label="Detected Named Entities",
        combine_adjacent=True,
        show_legend=True
    )

    detect_btn.click(
        fn=predict,
        inputs=textbox,
        outputs=output
    )

    gr.Examples(
        examples=[
            ["Ahmed works at Google in London."],
            ["Apple hired John yesterday."],
            ["Microsoft announced a new AI model in California."],
            ["Barack Obama was born in Hawaii."],
            ["Amazon opened a new office in Egypt."]
        ],
        inputs=textbox
    )

    gr.Markdown(
        """
        ---
        ### AI Project

        **Named Entity Recognition using DistilBERT**

        Dataset: **CoNLL-2003**

        Developed using **PyTorch • HuggingFace • Gradio**
        """
    )

demo.launch()