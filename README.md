# GameRS: Personalized Game Recommendation System

**GameRS** is a machine learning-powered game recommendation system that helps users discover new games tailored to their preferences. Built with **Streamlit**, it integrates **content-based filtering**, **cosine similarity**, and a **LLaMA-powered chatbot** to deliver a smart, engaging, and personalized user experience.

## Features

- **Search and Explore:** Look up your favorite games from a dataset of 5,000+ titles.
- **Smart Recommendations:** Recommends games using cosine similarity based on user reviews and metadata.
- **User Satisfaction Index for Games (USIG):** Uses a custom scoring algorithm that weighs ratings, genre overlap, and platform similarity.
- **AI Chatbot Integration:** Interact with a LLaMA-powered chatbot to ask questions about the recommended games.
- **Game Media Preview:** View trailers and screenshots of each game within the app.
- **Modern UI:** Built with Streamlit and styled using custom CSS for an intuitive and responsive interface.

## Project Structure

```
main
├─ data
│  ├─ cosine_sim500.pkl
│  ├─ cosine_sim5000.pkl
│  ├─ games_data500.csv
│  ├─ games_data500.pkl
│  ├─ games_data5000.csv
│  ├─ games_data5000.pkl
│  ├─ k_means5000.pkl
│  ├─ svd_matrix500.pkl
│  └─ svd_matrix5000.pkl
├─ GameRS.py
├─ img
│  └─ logo.png
├─ notebooks
│  ├─ cosine_sim.ipynb
│  ├─ k_means.ipynb
│  └─ svd_matrix.ipynb
├─ README.md
└─ static
   └─ style.css

```

## Tech Stack

- Frontend: Streamlit + HTML/CSS
- Backend: Python
- Recommendation Engine: Cosine Similarity, Content-Based Filtering
- AI Assistant: LLaMA (local chatbot integration)
- API Integration: [RAWG Video Games Database API](https://rawg.io/apidocs)

## Installation

1. Clone the Repository

```bash
git clone https://github.com/your-username/GameRS.git
cd GameRS
```

2. Run the Application

```bash
streamlit run GameRS.py
```

## Recommendation Logic

- Cosine Similarity: Used to compute similarity between games based on user reviews.
- USIG Scoring:
  - Normalized Rating Score
  - Genre Overlap Index
  - Platform Similarity Score
- Combines these metrics to rank and recommend the most relevant games for the user.

## Future Improvements

- Add multiplayer and cross-platform filters
- Support user login and profile tracking
- Integrate collaborative filtering for hybrid models
- Deploy chatbot on cloud or HuggingFace Spaces


## Acknowledgements

- RAWG API: https://rawg.io/
- LLaMA by Meta AI: https://ai.meta.com/llama/
- Streamlit: https://streamlit.io/