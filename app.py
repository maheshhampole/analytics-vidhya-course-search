import requests
pip install streamlit
from bs4 import BeautifulSoup
import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import logging
import concurrent.futures
from typing import List, Dict
import os

class CourseListingScraper:
    def __init__(self):
        self.setup_selenium()
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_selenium(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        self.driver = webdriver.Chrome(options=chrome_options)

    def get_course_urls(self, main_url: str) -> List[str]:
        """Extract all course URLs from the main listing page"""
        try:
            self.driver.get(main_url)
            # Wait for the page to load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".course-card, .card, article"))
            )

            # Scroll to load all content
            SCROLL_PAUSE_TIME = 2
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            while True:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(SCROLL_PAUSE_TIME)

                # Try to click "Load More" if it exists
                try:
                    load_more = self.driver.find_element(By.CSS_SELECTOR,
                        "[class*='load-more'], [class*='loadMore'], .show-more")
                    self.driver.execute_script("arguments[0].click();", load_more)
                    time.sleep(1)
                except:
                    pass

                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            # Get page source and parse with BeautifulSoup
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            # Enhanced selectors for course links
            course_links = []
            selectors = [
                'a[href*="/course"]',
                '.course-card a',
                '.course-listing a',
                '.course-title a',
                'article a',
                '.card a',
                '[class*="course"] a'
            ]

            for selector in selectors:
                links = soup.select(selector)
                if links:
                    for link in links:
                        href = link.get('href')
                        if href and ('/course' in href or '/learn' in href):
                            if not href.startswith('http'):
                                if href.startswith('/'):
                                    href = f"https://courses.analyticsvidhya.com{href}"
                                else:
                                    href = f"https://courses.analyticsvidhya.com/{href}"
                            course_links.append(href)

            # Remove duplicates while preserving order
            course_links = list(dict.fromkeys(course_links))

            self.logger.info(f"Found {len(course_links)} course URLs")
            return course_links

        except Exception as e:
            self.logger.error(f"Error extracting course URLs: {str(e)}")
            return []
        finally:
            self.driver.quit()

class CourseScraper:
    def __init__(self, url: str):
        self.url = url
        self.setup_selenium()
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_selenium(self):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1920,1080')
        self.driver = webdriver.Chrome(options=chrome_options)

    def _expand_curriculum(self):
        """Expand all curriculum sections"""
        try:
            # Find and click all expand buttons
            expand_buttons = self.driver.find_elements(By.CSS_SELECTOR,
                "[class*='expand'], [class*='toggle'], .show-more, .view-more")
            for button in expand_buttons:
                try:
                    self.driver.execute_script("arguments[0].click();", button)
                    time.sleep(0.5)
                except:
                    continue
        except Exception as e:
            self.logger.warning(f"Error expanding curriculum: {str(e)}")

    def scrape(self) -> Dict:
        try:
            self.driver.get(self.url)
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Expand any collapsed sections
            self._expand_curriculum()
            time.sleep(2)

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            # Enhanced course data extraction
            course_data = {
                'title': self._get_text(soup, [
                    'h1',
                    '.course-title',
                    '.title',
                    '.course-heading',
                    '.heading',
                    '[class*="title"]'
                ]),
                'description': self._get_text(soup, [
                    '.course-description',
                    '.description',
                    '#description',
                    '.about-course',
                    '.course-about',
                    '[class*="description"]',
                    '[class*="about"]',
                    '.course-info p'
                ]),
                'instructor': {
                    'name': self._get_text(soup, [
                        '.instructor-name',
                        '.author-name',
                        '.teacher-name',
                        '.faculty-name',
                        '[class*="instructor"] h3',
                        '[class*="author"] h3',
                        '[class*="teacher"] h3'
                    ]),
                    'bio': self._get_text(soup, [
                        '.instructor-bio',
                        '.author-bio',
                        '.teacher-bio',
                        '[class*="instructor"] p',
                        '[class*="author"] p',
                        '[class*="teacher"] p'
                    ]),
                    'title': self._get_text(soup, [
                        '.instructor-title',
                        '.author-title',
                        '.teacher-title',
                        '[class*="instructor"] .title',
                        '[class*="author"] .title'
                    ])
                },
                'details': {
                    'duration': self._get_text(soup, [
                        '.content-length',
                        '.duration',
                        '.course-duration',
                        '.length',
                        '[class*="duration"]',
                        '[class*="length"]'
                    ]),
                    'level': self._get_text(soup, [
                        '.course-level',
                        '.level',
                        '.difficulty',
                        '[class*="level"]',
                        '[class*="difficulty"]'
                    ]),
                    'category': self._get_text(soup, [
                        '.course-category',
                        '.category',
                        '[class*="category"]',
                        '[class*="topic"]'
                    ]),
                    'language': self._get_text(soup, [
                        '.course-language',
                        '.language',
                        '[class*="language"]'
                    ])
                },
                'rating': {
                    'score': self._get_text(soup, [
                        '.course-rating',
                        '.rating',
                        '.stars',
                        '[class*="rating"]',
                        '[class*="stars"]'
                    ]),
                    'count': self._get_text(soup, [
                        '.rating-count',
                        '.reviews-count',
                        '[class*="reviews"]',
                        '[class*="rating-count"]'
                    ])
                },
                'pricing': {
                    'current_price': self._get_text(soup, [
                        '.course-price',
                        '.price',
                        '.current-price',
                        '[class*="price"]'
                    ]),
                    'original_price': self._get_text(soup, [
                        '.original-price',
                        '.old-price',
                        '.list-price',
                        '[class*="original"]',
                        '.strikethrough'
                    ]),
                    'discount': self._get_text(soup, [
                        '.discount',
                        '.savings',
                        '[class*="discount"]',
                        '[class*="save"]'
                    ])
                },
                'curriculum': self._get_curriculum(soup),
                'requirements': self._get_requirements(soup),
                'learning_objectives': self._get_learning_objectives(soup),
                'url': self.url,
                'last_updated': self._get_text(soup, [
                    '.last-updated',
                    '.update-date',
                    '[class*="updated"]',
                    '[class*="date"]'
                ])
            }

            return course_data

        except Exception as e:
            self.logger.error(f"Error scraping course: {str(e)}")
            return None
        finally:
            self.driver.quit()

    def _get_curriculum(self, soup) -> List[Dict]:
        """Extract curriculum information"""
        curriculum = []

        section_selectors = [
            '.curriculum-section',
            '.syllabus-section',
            '.course-content',
            '.course-curriculum',
            '[class*="curriculum"]',
            '[class*="syllabus"]',
            '.course-lessons'
        ]

        for selector in section_selectors:
            sections = soup.select(selector)
            if sections:
                for section in sections:
                    section_data = {
                        'title': self._get_text(section, [
                            '.section-title',
                            '.chapter-title',
                            'h3',
                            'h4'
                        ]),
                        'lessons': []
                    }

                    # Find lessons within section
                    lessons = section.select('.lesson, .lecture, .topic, li')
                    for lesson in lessons:
                        lesson_data = {
                            'title': self._get_text(lesson, [
                                '.lesson-title',
                                '.lecture-title',
                                'h4',
                                'h5',
                                'strong'
                            ]),
                            'duration': self._get_text(lesson, [
                                '.duration',
                                '.length',
                                '[class*="duration"]',
                                '[class*="length"]'
                            ])
                        }
                        if lesson_data['title']:
                            section_data['lessons'].append(lesson_data)

                    curriculum.append(section_data)
                break

        return curriculum

    def _get_requirements(self, soup) -> List[str]:
        """Extract course requirements"""
        requirements = []
        requirement_selectors = [
            '.requirements li',
            '.prerequisites li',
            '[class*="requirement"] li',
            '[class*="prerequisite"] li',
            '.course-requirements li'
        ]

        for selector in requirement_selectors:
            items = soup.select(selector)
            if items:
                requirements.extend([item.text.strip() for item in items if item.text.strip()])

        return requirements

    def _get_learning_objectives(self, soup) -> List[str]:
        """Extract learning objectives"""
        objectives = []
        objective_selectors = [
            '.learning-objective li',
            '.course-objective li',
            '.outcomes li',
            '[class*="learn"] li',
            '[class*="objective"] li',
            '.what-you-learn li'
        ]

        for selector in objective_selectors:
            items = soup.select(selector)
            if items:
                objectives.extend([item.text.strip() for item in items if item.text.strip()])

        return objectives

    def _get_text(self, soup, selectors, default=''):
        """Extract text using multiple selectors"""
        for selector in selectors:
            try:
                elements = soup.select(selector)
                if elements:
                    # Join multiple elements if found
                    text = ' '.join(elem.text.strip() for elem in elements if elem.text.strip())
                    # Clean up the text
                    text = ' '.join(text.split())  # Remove extra whitespace
                    return text if text else default
            except Exception:
                continue
        return default

def save_courses(courses_data: List[Dict], output_dir='scraped_courses'):
    """Save scraped data to JSON and CSV formats"""
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed JSON
    json_path = os.path.join(output_dir, 'courses_detailed.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(courses_data, f, ensure_ascii=False, indent=2)

    # Save simplified CSV
    csv_data = []
    for course in courses_data:
        flat_course = {
            'title': course['title'],
            'url': course['url'],
            'description': course['description'],
            'instructor_name': course['instructor']['name'],
            'instructor_title': course['instructor']['title'],
            'duration': course['details']['duration'],
            'level': course['details']['level'],
            'category': course['details']['category'],
            'language': course['details']['language'],
            'rating_score': course['rating']['score'],
            'rating_count': course['rating']['count'],
            'current_price': course['pricing']['current_price'],
            'original_price': course['pricing']['original_price'],
            'discount': course['pricing']['discount'],
            'last_updated': course['last_updated'],
            'requirements': '; '.join(course['requirements']),
            'learning_objectives': '; '.join(course['learning_objectives'])
        }
        csv_data.append(flat_course)

    csv_path = os.path.join(output_dir, 'courses_simplified.csv')
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)

    print(f"\nScraped data saved to:")
    print(f"- Detailed JSON: {json_path}")
    print(f"- Simplified CSV: {csv_path}")


def main():
    # URL of the main course listing page
    main_url = "https://courses.analyticsvidhya.com/pages/all-free-courses"

    print("Starting course scraper...")

    # First, get all course URLs
    listing_scraper = CourseListingScraper()
    course_urls = listing_scraper.get_course_urls(main_url)

    if not course_urls:
        print("No course URLs found!")
        return

    print(f"Found {len(course_urls)} courses to scrape")

    # Scrape each course
    all_courses_data = []
    for url in course_urls:
        try:
            print(f"Scraping: {url}")
            scraper = CourseScraper(url)
            course_data = scraper.scrape()
            if course_data:
                all_courses_data.append(course_data)
                print(f"✓ Successfully scraped: {url}")
            else:
                print(f"✗ Failed to scrape: {url}")
        except Exception as e:
            print(f"✗ Error scraping {url}: {str(e)}")

    # Save results
    if all_courses_data:
        save_courses(all_courses_data)
        print(f"\nSuccessfully scraped {len(all_courses_data)} out of {len(course_urls)} courses")
    else:
        print("\nNo courses were successfully scraped")

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import sentence_transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import json
import torch
from typing import List, Dict, Tuple
import re

class CourseSearchEngine:
    def __init__(self, model_name: str = 'paraphrase-MiniLM-L6-v2'):
        """Initialize the search engine with a sentence transformer model"""
        self.model = SentenceTransformer(model_name)
        self.courses_df = None
        self.course_embeddings = None

    def load_courses(self, json_path: str):
        """Load and prepare course data"""
        with open(json_path, 'r', encoding='utf-8') as f:
            courses = json.load(f)

        # Convert to DataFrame
        self.courses_df = pd.DataFrame(courses)

        # Prepare search content for each course
        self.courses_df['search_content'] = self.courses_df.apply(self._prepare_search_content, axis=1)

        # Generate embeddings
        self.course_embeddings = self.model.encode(
            self.courses_df['search_content'].tolist(),
            convert_to_tensor=True
        )

    def _prepare_search_content(self, row: pd.Series) -> str:
        """Combine relevant course information for searching"""
        content_parts = [
            str(row.get('title', '')),
            str(row.get('description', '')),
            str(row.get('instructor', {}).get('name', '')),
            str(row.get('content_length', ''))
        ]
        return ' '.join([part for part in content_parts if part and part != 'nan'])

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for courses based on query"""
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Calculate similarity scores
        similarities = cosine_similarity(
            query_embedding.cpu().numpy().reshape(1, -1),
            self.course_embeddings.cpu().numpy()
        )[0]

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            course = self.courses_df.iloc[idx].to_dict()
            course['similarity_score'] = float(similarities[idx])
            results.append(course)

        return results

    def get_course_suggestions(self, query: str) -> List[str]:
        """Get course title suggestions based on partial query"""
        if not query:
            return []

        query_lower = query.lower()
        suggestions = []

        for title in self.courses_df['title']:
            if query_lower in title.lower():
                suggestions.append(title)

        return suggestions[:5]  # Limit to top 5 suggestions

# Streamlit UI
def create_streamlit_app():
    st.set_page_config(
        page_title="Analytics Vidhya Course Search",
        page_icon="🎓",
        layout="wide"
    )

    # Initialize search engine
    @st.cache_resource
    def load_search_engine():
        engine = CourseSearchEngine()
        engine.load_courses('/content/scraped_courses/courses_detailed.json')
        return engine

    search_engine = load_search_engine()

    # UI Elements
    st.title("🎓 Analytics Vidhya Course Search")
    st.markdown("""
    Find the perfect free course from Analytics Vidhya's collection.
    Enter your interests, topics, or skills you want to learn!
    """)

    # Search input with auto-complete
    query = st.text_input(
        "Search courses",
        key="search_input",
        placeholder="Enter topics, skills, or keywords..."
    )

    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

    # Search button
    search_clicked = st.button("Search", type="primary")

    # Show results
    if search_clicked and query:
        results = search_engine.search(query, top_k=top_k)

        if results:
            st.subheader(f"Found {len(results)} relevant courses")

            for i, course in enumerate(results, 1):
                score = course['similarity_score']
                relevance = int(score * 100)

                with st.expander(
                    f"{i}. {course['title']} (Relevance: {relevance}%)",
                    expanded=i == 1
                ):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**Description:**\n{course['description']}")
                        st.markdown(f"**Instructor:** {course['instructor']['name']}")

                    with col2:
                        st.markdown(f"**Duration:** {course['content_length']}")
                        if course.get('rating'):
                            st.markdown(f"**Rating:** {course['rating']}")
                        st.markdown(f"**Price:** {course.get('price', 'Free')}")

                    if course.get('url'):
                        st.markdown(f"[Go to Course]({course['url']})")
        else:
            st.warning("No courses found matching your search.")

    # Add footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Sentence Transformers and Streamlit | "
        "Data from Analytics Vidhya's Free Courses"
    )

if __name__ == "__main__":
    create_streamlit_app()


