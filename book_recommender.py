import streamlit as st
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def fetch_books(query, max_results=40):
    search_url = f"https://openlibrary.org/search.json?q={query}&limit={max_results}"
    response = requests.get(search_url)
    
    if response.status_code != 200:
        st.error("Failed to fetch data from Open Library API.")
        return None
    
    data = response.json()
    books = []
    
    for doc in data.get('docs', []):
        title = doc.get('title', 'Unknown Title')
        cover_id = doc.get('cover_i')
        image = f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else None
        key = doc.get('key', '')
        
        description = ''
        if key:
            works_url = f"https://openlibrary.org{key}.json"
            works_response = requests.get(works_url)
            if works_response.status_code == 200:
                works_data = works_response.json()
                description = works_data.get('description', '')
                if isinstance(description, dict):
                    description = description.get('value', '')

        books.append({
            'title': title,
            'description': description,
            'image': image
        })
    
    return pd.DataFrame(books)

def get_recommendations(book_title, books_df):
    if book_title not in books_df['title'].values:
        return "Book title not found in the dataset", None
    
    books_df['description'] = books_df['description'].fillna('')
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books_df['description'])
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    idx = books_df[books_df['title'] == book_title].index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    
    book_indices = [i[0] for i in sim_scores]
    
    recommended_books = books_df.iloc[book_indices][['title', 'image']]
    return recommended_books, None

def main():
    st.markdown("""
        <style>
        div[data-testid="column"] {
            padding: 20px;
            margin: 15px;
            background-color: #2a2a2a;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .book-image-container {
            height: 220px;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }
        .book-image-container img {
            max-height: 100%;
            max-width: 100%;
            object-fit: contain;
        }
        .stImageCaption {
            font-size: 18px !important;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
            margin-top: 15px;
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Book Recommender System")
    
    st.sidebar.header("Search for a Book")
    book_title = st.sidebar.text_input("Enter a Book Title")
    
    if st.sidebar.button("Recommend"):
        if book_title:
            with st.spinner("Fetching book data from Open Library..."):
                books_df = fetch_books(book_title)
            
            if books_df is None or books_df.empty:
                st.error("No results found or failed to fetch books.")
            else:
                recommended_books, error = get_recommendations(book_title, books_df)
                
                if isinstance(recommended_books, str):
                    st.error(recommended_books)
                else:
                    st.subheader("Recommended Books:")
                    num_cols = 5
                    num_books = len(recommended_books)
                    num_rows = (num_books + num_cols - 1) // num_cols

                    for row in range(num_rows):
                        cols = st.columns(num_cols)
                        for col in range(num_cols):
                            book_idx = row * num_cols + col
                            if book_idx < num_books:
                                with cols[col]:
                                    book = recommended_books.iloc[book_idx]
                                    if book['image']:
                                        st.markdown(
                                            f"""
                                            <div class="book-image-container">
                                                <img src="{book['image']}" alt="{book['title']}">
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )
                                        st.caption(book['title'])
                                    else:
                                        st.write("No image available")
                                        st.caption(book['title'])
        else:
            st.warning("Please enter a valid book title.")

if __name__ == '__main__':
    main()