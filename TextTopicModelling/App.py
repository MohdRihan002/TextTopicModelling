import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from PyPDF2 import PdfReader
from preprocessing import clean_text
from topic_model import perform_topic_modeling

import csv
from datetime import datetime

st.set_page_config(
    page_title="Text Topic Modelling",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Text Topic Modelling System")
st.write(
    "Discover hidden topics from your text documents "
    "using Natural Language Processing."
)


st.sidebar.header("⚙️ Project Settings")

# 1. Number of Topics
num_topics = st.sidebar.slider(
    "Number of Topics",
    min_value=2,
    max_value=10,
    value=3
)

# 2. Topic Modelling Algorithm
algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["LDA", "NMF"]
)

# 3. Number of Keywords
num_keywords = st.sidebar.slider(
    "Keywords per Topic",
    min_value=5,
    max_value=20,
    value=10
)

# 4. Minimum Word Length
min_word_length = st.sidebar.slider(
    "Minimum Word Length",
    min_value=2,
    max_value=8,
    value=3
)

# 5. Word Cloud
show_wordcloud = st.sidebar.checkbox(
    "Show Word Cloud",
    value=True
)

# 6. Charts
show_charts = st.sidebar.checkbox(
    "Show Charts",
    value=True
)

st.subheader("Enter Text")

user_text = st.text_area(
    "Paste your text here. Separate documents with blank lines.",
    height=200
)

st.subheader("Upload Documents")

uploaded_files = st.file_uploader(
    "Upload TXT, PDF or CSV files",
    type=["txt", "pdf", "csv"],
    accept_multiple_files=True
)

documents = []
# Create upload storage folder
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Read pasted text
if user_text.strip():
    documents.extend([
        part.strip()
        for part in user_text.split("\n\n")
        if part.strip()
    ])

# Read uploaded files
for file in uploaded_files:
        # Save uploaded original file
    file_path = UPLOAD_DIR / Path(file.name).name
    file_path.write_bytes(file.getvalue())

    if file.name.endswith(".txt"):
        content = file.read().decode(
            "utf-8", errors="ignore"
        )
        documents.extend([
            part.strip()
            for part in content.split("\n\n")
            if part.strip()
        ])

    elif file.name.endswith(".pdf"):
        reader = PdfReader(file)

        for page in reader.pages:
            content = page.extract_text() or ""

            if content.strip():
                documents.append(content)

    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)

        text_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns

        if len(text_columns) > 0:
            selected_column = text_columns[0]

            documents.extend(
                df[selected_column]
                .dropna()
                .astype(str)
                .tolist()
            )

st.write("Total documents:", len(documents))

if st.button("Analyze Documents"):

    if len(documents) < 2:
        st.error(
            "Please provide at least 2 text documents."
        )

    else:
        cleaned_documents = [
            clean_text(doc) for doc in documents
        ]

        cleaned_documents = [
            doc for doc in cleaned_documents if doc
        ]

        if len(cleaned_documents) < 2:
            st.error("Please provide more meaningful text.")

        else:
            
            try:
                topics, doc_topics = perform_topic_modeling(
                    cleaned_documents,
                    num_topics,
                    algorithm,
                    num_keywords
                )

                # Word Cloud
                if show_wordcloud:
                    st.subheader("☁️ Word Cloud Visualization")

                    combined_text = " ".join(cleaned_documents)

                    wordcloud = WordCloud(
                        width=1200,
                        height=500,
                        background_color="white",
                        max_words=100,
                        collocations=False
                    ).generate(combined_text)

                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")

                    st.pyplot(fig)
                    plt.close(fig)

                st.success("Topic modelling completed!")

                # Identified Topics
                st.subheader("Identified Topics")

                for topic in topics:
                    st.markdown(f"### Topic {topic['Topic']}")
                    st.write(", ".join(topic["Keywords"]))

                # Charts
                if show_charts:

                    st.subheader("Document Topic Distribution")

                    chart_data = pd.DataFrame(
                        doc_topics,
                        columns=[
                            f"Topic {i+1}"
                            for i in range(doc_topics.shape[1])
                        ]
                    )

                    st.subheader("🥧 Overall Topic Percentage")

                    average_topic_scores = chart_data.mean()
                    topic_percentage = average_topic_scores * 100

                    percentage_df = pd.DataFrame({
                        "Topic": topic_percentage.index,
                        "Percentage": topic_percentage.values.round(2)
                    })

                    st.dataframe(
                        percentage_df,
                        use_container_width=True
                    )

                    fig, ax = plt.subplots(figsize=(8, 6))

                    ax.pie(
                        topic_percentage,
                        labels=topic_percentage.index,
                        autopct="%1.1f%%",
                        startangle=90
                    )

                    ax.set_title("Overall Topic Distribution")

                    st.pyplot(fig)
                    plt.close(fig)

                    # Bar Chart
                    st.bar_chart(chart_data)

                # Document-wise Results (outside show_charts)
                st.subheader("Document-wise Results")

                results = []

                for i, scores in enumerate(doc_topics):
                    dominant_topic = scores.argmax() + 1

                    results.append({
                        "Document": i + 1,
                        "Dominant Topic": dominant_topic,
                        "Confidence (%)": round(
                            scores.max() * 100, 2
                        )
                    })

                result_df = pd.DataFrame(results)

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                csv = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Results",
                    data=csv,
                    file_name="topic_results.csv",
                    mime="text/csv"
                )

            except ValueError as error:
                st.error(str(error))

            except Exception as error:
                st.error(f"An error occurred: {error}")


                
# FEEDBACK SECTION
st.divider()
st.subheader("📝 Share Your Feedback")

with st.form("feedback_form"):

    name = st.text_input("Your Name (Optional)")

    rating = st.slider(
        "Rate our project",
        min_value=1,
        max_value=5,
        value=5
    )

    feedback = st.text_area(
        "Write your feedback",
        placeholder="Tell us what you liked or what we can improve..."
    )

    submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        if not feedback.strip():
            st.warning("Please write your feedback.")
        else:
            feedback_dir = Path("data")
            feedback_dir.mkdir(exist_ok=True)

            feedback_file = feedback_dir / "feedback.csv"
            file_exists = feedback_file.exists()

            with open(
                feedback_file,
                "a",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow([
                        "Date", "Name", "Rating", "Feedback"
                    ])

                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    name,
                    rating,
                    feedback
                ])

            st.success("Thank you! Your feedback has been saved.")