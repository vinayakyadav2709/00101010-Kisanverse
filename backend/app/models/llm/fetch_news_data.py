# fetch_news_data.py
import json
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import quote_plus, urljoin

# import uuid # Use URL as ID for simplicity now
import chromadb  # Vector DB
from sentence_transformers import SentenceTransformer  # For embeddings

# Import config directly
import models.llm.config as config

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36"
}

# --- Initialize ChromaDB and Embedding Model ---
# These are initialized globally to be reused if the script is imported
# However, the main execution block handles the core logic
chroma_client: Optional[chromadb.ClientAPI] = None
collection: Optional[chromadb.Collection] = None
embedding_model: Optional[SentenceTransformer] = None


def initialize_vector_db_and_model():
    """Initializes ChromaDB client, collection, and embedding model."""
    global chroma_client, collection, embedding_model
    if collection and embedding_model:
        log.info("ChromaDB and embedding model already initialized.")
        return True
    try:
        log.info(
            f"Initializing ChromaDB client with persistent path: {config.CHROMA_DB_PATH}"
        )
        # Ensure the directory exists before creating the client
        os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

        log.info(f"Getting/Creating Chroma collection: {config.CHROMA_COLLECTION_NAME}")
        # Get or create the collection. Let Chroma handle default embedding function setup initially.
        # We provide embeddings explicitly during add.
        collection = chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME
        )
        log.info(
            f"Collection '{config.CHROMA_COLLECTION_NAME}' loaded/created ({collection.count()} items existing)."
        )

        log.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        log.info("Embedding model loaded successfully.")
        return True

    except Exception as e:
        log.critical(
            f"Failed to initialize ChromaDB or Embedding Model: {e}", exc_info=True
        )
        return False


def scrape_google_news(queries: List[str], lookback_days: int) -> List[Dict]:
    """Scrapes Google News search results, returning list of news dicts."""
    log.info(
        f"Scraping Google News for {len(queries)} queries (Lookback: {lookback_days} days)..."
    )
    all_items = []
    processed_urls = set()
    # Set cutoff date in UTC
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    for query in queries:
        search_url = config.GOOGLE_NEWS_URL.format(query=quote_plus(query))
        log.info(f"Fetching: {search_url}")
        try:
            response = requests.get(
                search_url, headers=HEADERS, timeout=config.SCRAPER_TIMEOUT_SECONDS
            )
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            soup = BeautifulSoup(
                response.text, "lxml"
            )  # Use lxml for speed if installed

            articles = soup.find_all("article")
            log.info(f"Found {len(articles)} potential articles for query '{query}'")

            for article in articles:
                headline_tag = (
                    article.find("a", class_="JtKRv")
                    or article.find("h3")
                    or article.find("h4")
                )
                link_tag = article.find("a", href=True)
                source_tag = article.find("div", class_="vr1PYe") or article.find(
                    "div", class_="wfBUxe"
                )
                time_tag = article.find("time", datetime=True)
                # Try to get description snippet
                desc_tag = article.find("div", class_="Y3v8qd") or article.find(
                    "span", class_="rQMQod"
                )

                if headline_tag and link_tag:
                    headline = headline_tag.get_text(strip=True)
                    raw_url = link_tag["href"]
                    # Resolve relative URLs from Google News search results page
                    url = urljoin("https://news.google.com/", raw_url.lstrip("."))

                    # Skip if URL already processed in this run
                    if url in processed_urls:
                        continue

                    source = (
                        source_tag.get_text(strip=True)
                        if source_tag
                        else "Unknown Source"
                    )
                    published_date: Optional[datetime] = None
                    published_date_iso: str = datetime.now(
                        timezone.utc
                    ).isoformat()  # Default to now if parsing fails
                    description = desc_tag.get_text(strip=True) if desc_tag else ""

                    if time_tag and time_tag.has_attr("datetime"):
                        try:
                            # Parse ISO 8601 format, ensure timezone aware (UTC)
                            published_date = datetime.fromisoformat(
                                time_tag["datetime"].replace("Z", "+00:00")
                            )
                            published_date_iso = published_date.isoformat()
                        except ValueError:
                            log.warning(
                                f"Could not parse datetime: {time_tag['datetime']} for headline: '{headline}'"
                            )
                            # Keep default published_date_iso (now)

                    # Filter by date (only if successfully parsed)
                    if published_date and published_date < cutoff_date:
                        log.debug(
                            f"Skipping old item (Published: {published_date_iso}): '{headline}'"
                        )
                        continue

                    # Add item if within lookback period or date couldn't be parsed
                    news_item = {
                        "date": published_date_iso,
                        "headline": headline,
                        "source": source,
                        "description": description,
                        "url": url,
                        "query_source": query,  # Track which query found this item
                    }
                    all_items.append(news_item)
                    processed_urls.add(url)

                else:
                    log.debug("Skipping block - missing headline or link tag.")

        except requests.exceptions.Timeout:
            log.error(f"Timeout error fetching query '{query}' from {search_url}")
        except requests.exceptions.RequestException as e:
            log.error(f"Network error for query '{query}': {e}")
        except Exception as e:
            log.error(f"Error processing query '{query}': {e}", exc_info=True)

        log.debug(f"Waiting {config.REQUEST_DELAY_SECONDS}s before next query...")
        time.sleep(config.REQUEST_DELAY_SECONDS)

    # Sort by date descending (newest first)
    all_items.sort(key=lambda x: x.get("date", ""), reverse=True)
    log.info(
        f"Scraping finished. Found {len(all_items)} unique news items within lookback period."
    )
    return all_items


def store_news_in_chroma(news_items: List[Dict]):
    """Generates embeddings and stores news items in ChromaDB if not already present, using URL as ID."""
    if not collection or not embedding_model:
        log.error(
            "ChromaDB collection or embedding model not initialized. Cannot store news."
        )
        return
    if not news_items:
        log.warning("No news items provided to store.")
        return

    log.info(
        f"Attempting to store/update {len(news_items)} scraped items in ChromaDB collection '{config.CHROMA_COLLECTION_NAME}'..."
    )
    added_count = 0
    updated_count = 0
    skipped_count = 0
    ids_to_upsert = []
    embeddings_to_upsert = []
    metadatas_to_upsert = []
    documents_to_upsert = []  # Text content for Chroma's potential use

    # Check existing IDs (URLs) in the collection to decide between add/update
    urls_to_check = [item.get("url") for item in news_items if item.get("url")]
    existing_ids = set()
    if urls_to_check:
        try:
            # Fetch existing items by ID (URL) - check only IDs, don't need full data yet
            existing_items_response = collection.get(
                ids=urls_to_check, include=[]
            )  # include=[] fetches only existence
            existing_ids = set(existing_items_response.get("ids", []))
            log.info(
                f"Checked {len(urls_to_check)} URLs against ChromaDB, {len(existing_ids)} already exist and will be updated/skipped if identical."
            )
        except Exception as e:
            log.error(
                f"Error checking existing items in ChromaDB: {e}. Proceeding with upsert, may overwrite.",
                exc_info=True,
            )
            # Continue, upsert will handle additions or overwrites

    for item in news_items:
        url = item.get("url")
        headline = item.get("headline")

        # Basic validation: Need URL (as ID) and Headline (for embedding)
        if not url or not headline:
            log.warning(
                f"Skipping item due to missing URL or headline: {item.get('headline', 'N/A URL')}"
            )
            skipped_count += 1
            continue

        # Use URL as the unique ID
        doc_id = url

        # Create metadata dictionary (ensure values are Chroma-compatible types: str, int, float, bool)
        metadata = {
            "source": str(item.get("source", "Unknown")),
            "date": str(item.get("date", "")),  # Store date string
            "url": url,
            "headline": headline,  # Also store headline in metadata for easy retrieval/display
            # Add description if available and not empty
        }
        desc = item.get("description")
        if desc:
            metadata["description"] = str(desc)

        # Text to embed (Combine headline and description for richer context)
        text_to_embed = headline
        if desc:
            text_to_embed += ". " + desc  # Simple concatenation

        # Generate embedding
        try:
            # Ensure model is loaded
            if not embedding_model:
                raise ValueError("Embedding model not loaded.")
            embedding = embedding_model.encode(text_to_embed).tolist()
        except Exception as e:
            log.error(f"Failed to generate embedding for item URL {doc_id}: {e}")
            skipped_count += 1
            continue

        # Add to lists for batch upsertion
        ids_to_upsert.append(doc_id)
        embeddings_to_upsert.append(embedding)
        metadatas_to_upsert.append(metadata)
        documents_to_upsert.append(text_to_embed)  # Store the text used for embedding

        if doc_id in existing_ids:
            updated_count += 1
        else:
            added_count += 1

    # Batch upsert into ChromaDB
    # Upsert will add new documents and update existing ones based on ID
    if ids_to_upsert:
        log.info(
            f"Upserting {len(ids_to_upsert)} items into Chroma collection '{config.CHROMA_COLLECTION_NAME}' ({added_count} new, {updated_count} updates)..."
        )
        try:
            collection.upsert(
                ids=ids_to_upsert,
                embeddings=embeddings_to_upsert,
                metadatas=metadatas_to_upsert,
                documents=documents_to_upsert,
            )
            log.info(
                f"Batch upsert successful. Collection now has {collection.count()} items."
            )
        except Exception as e:
            log.error(f"Error upserting batch to ChromaDB: {e}", exc_info=True)
            # Log which items might have failed if possible, though batch ops make this hard
    else:
        log.info("No valid items prepared for upsertion into ChromaDB.")

    log.info(
        f"Finished storing news. Added: {added_count}, Updated: {updated_count}, Skipped: {skipped_count}"
    )


if __name__ == "__main__":
    log.info("--- Starting News Vector Database Update ---")

    # Ensure data directory exists
    if not os.path.exists(config.DATA_DIR):
        log.info(f"Creating data directory: {config.DATA_DIR}")
        os.makedirs(config.DATA_DIR)

    # Initialize ChromaDB and Model
    if not initialize_vector_db_and_model():
        log.critical("Exiting script due to DB/Model initialization failure.")
        exit(1)  # Exit if critical components fail

    # Scrape Google News
    news_items = scrape_google_news(config.SEARCH_QUERIES, config.NEWS_MONTHS_TO_FETCH)

    # Store in ChromaDB
    if news_items:
        store_news_in_chroma(news_items)
    else:
        log.warning(
            "No news items were scraped from Google News. ChromaDB not updated."
        )

    log.info("--- Finished News Data Fetching and Storing ---")
