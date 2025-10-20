import os
import json
import numpy as np
import requests
import streamlit as st

# Import sklearn with error handling
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.error("Please ensure scikit-learn is installed. Check your requirements.txt file.")
    st.stop()

# --- Configuration ---
OPENROUTER_API_KEY = "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"  # hardcoded per user request
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-3.5-turbo"
# Use relative paths for deployment compatibility
DATA_PATH = "titan_chatbot_data.json"
FAQ_PDF_PATH = "Titan_Company_FAQ.pdf"
MAX_SNIPPETS = 5
MAX_SNIPPET_CHARS = 1000

# --- Utilities ---
def get_api_key() -> str:
	# Priority: hardcoded variable, Streamlit secrets, environment
	key = OPENROUTER_API_KEY
	if key and key != "sk-or-v1-REPLACE_ME":
		return key
	key = st.secrets.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
	return key

@st.cache_resource
def load_artifacts(path: str = DATA_PATH):
	try:
		with open(path, 'r', encoding='utf-8') as f:
			data = json.load(f)
		pages = data['pages']
		vocab = data['vocabulary']
		matrix = np.array(data['tfidf_matrix'], dtype=float)
		return pages, vocab, matrix
	except FileNotFoundError:
		st.error(f"Required file not found: {path}")
		st.error("Please ensure titan_chatbot_data.json is uploaded to your Streamlit app.")
		st.stop()
	except Exception as e:
		st.error(f"Error loading data: {e}")
		st.stop()

@st.cache_resource
def build_vectorizer(vocabulary: dict):
	# CountVectorizer with fixed vocab for query encoding; no fitting needed
	return CountVectorizer(vocabulary=vocabulary)


def _safe_read_faq_pdf(pdf_path: str) -> list:
	"""Return a list of dicts [{'page_number': int, 'text': str, 'source': 'faq_pdf'}]."""
	pages = []
	try:
		# Prefer modern 'pypdf' if available; otherwise fall back to 'PyPDF2'
		try:
			import pypdf as pdf_backend  # type: ignore
		except Exception:
			import PyPDF2 as pdf_backend  # type: ignore
		with open(pdf_path, 'rb') as f:
			reader = pdf_backend.PdfReader(f)
			for idx, page in enumerate(reader.pages, start=1):
				# Both backends support extract_text(); handle None safely
				text = (page.extract_text() or '').strip()
				pages.append({'page_number': idx, 'text': text, 'source': 'faq_pdf'})
	except Exception as e:
		st.warning(f"FAQ PDF couldn't be processed ({e}). The bot will use only the JSON corpus.")
	return pages


@st.cache_resource
def load_faq_artifacts(pdf_path: str, _vectorizer: CountVectorizer):
	"""Load FAQ PDF pages and build a TF-IDF matrix aligned to existing vocabulary."""
	faq_pages = _safe_read_faq_pdf(pdf_path)
	if not faq_pages:
		return [], None
	texts = [(p.get('text') or '') for p in faq_pages]
	counts = _vectorizer.transform(texts)
	# Fit TF-IDF on FAQ pages only; this is separate from the annual report matrix
	tfidf = TfidfTransformer(norm='l2', use_idf=True)
	faq_matrix = tfidf.fit_transform(counts)
	return faq_pages, faq_matrix


def retrieve_context(query: str, vectorizer: CountVectorizer, tfidf_matrix: np.ndarray, pages: list, k: int = MAX_SNIPPETS) -> list:
	query_counts = vectorizer.transform([query])
	query_vec = normalize(query_counts, norm='l2')
	scores = cosine_similarity(query_vec, tfidf_matrix)[0]
	idx_sorted = np.argsort(scores)[::-1][:k]
	selected = []
	for i in idx_sorted:
		selected.append({
			'page_number': pages[i].get('page_number'),
			'text': (pages[i].get('text') or '')[:MAX_SNIPPET_CHARS],
			'score': float(scores[i]),
			'source': pages[i].get('source', 'annual_report')
		})
	return selected


def call_openrouter(api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
	headers = {
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json",
		# Optional but recommended headers
		"HTTP-Referer": "https://localhost",  # replace with your deployed URL if any
		"X-Title": "Titan Company FAQ Chatbot",
	}
	payload = {
		"model": model,
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
		"temperature": 0.2,
		"max_tokens": 500,
	}
	resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
	resp.raise_for_status()
	data = resp.json()
	return data["choices"][0]["message"]["content"].strip()


def build_prompt(question: str, snippets: list) -> tuple[str, str]:
	snippet_text = "\n\n".join([f"[{s.get('source','annual_report').replace('_',' ')} | Page {s['page_number']} | score {s['score']:.3f}]\n{s['text']}" for s in snippets])
	system_prompt = (
		"You are a helpful assistant answering questions using ONLY the provided context from the Titan Company "
		"Annual Report (FY 2023-24) and the Titan Company FAQ PDF document. Quote figures cautiously and avoid fabricating details."
	)
	user_prompt = (
		f"Context:\n{snippet_text}\n\n"
		f"Question: {question}\n\n"
		"If the answer is not present in the context, say you couldn't find it explicitly and suggest the most relevant sections."
	)
	return system_prompt, user_prompt

# --- UI ---
st.set_page_config(page_title='Titan Company FAQ Chatbot', page_icon='⌚', layout='wide')

# Add custom CSS for full width text
st.markdown("""
<style>
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
    .stText {
        width: 100%;
    }
    .stMarkdown {
        width: 100%;
    }
    .element-container {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title('⌚ Titan Company Annual Report FAQ Chatbot')
st.caption('Lightweight RAG: TF‑IDF retrieval + OpenRouter GPT for answers.')

TOP_FAQ_QUESTIONS = [
	"How does Titan ensure product quality?",
	"What is Titan's approach to innovation?",
	"How does Titan maintain market leadership?",
	"How does Titan approach digital transformation?",
	"How does Titan manage risk across operations?",
	"What are Titan's main business segments?",
]

with st.sidebar:
	st.subheader('Settings')
	st.text_input('Override API key (optional)', type='password', key='override_key')
	st.number_input('Snippets (k)', min_value=1, max_value=10, value=MAX_SNIPPETS, key='k')
	st.number_input('Max snippet chars', min_value=200, max_value=2000, value=MAX_SNIPPET_CHARS, step=100, key='max_chars')
	
	# Add some spacing
	st.markdown("---")
	st.markdown("**Note:** Text content now uses full width for better readability.")

if st.session_state.get('override_key'):
	OPENROUTER_API_KEY = st.session_state['override_key']
if st.session_state.get('k'):
	MAX_SNIPPETS = int(st.session_state['k'])
if st.session_state.get('max_chars'):
	MAX_SNIPPET_CHARS = int(st.session_state['max_chars'])

pages, vocab, matrix = load_artifacts(DATA_PATH)
# Mark source on annual report pages for clarity in UI/prompts
for p in pages:
	p['source'] = 'annual_report'
vectorizer = build_vectorizer(vocab)
faq_pages, faq_matrix = load_faq_artifacts(FAQ_PDF_PATH, vectorizer)

# Initialize selected question from session state
selected_q = st.session_state.get('selected_question', '')
query = st.text_input('Your question', value=selected_q, placeholder="e.g., What are Titan's key business segments?", key='question')

# Track if a question has been answered
question_answered = st.session_state.get('question_answered', False)

col1, col2 = st.columns([4, 1])
with col1:
	if st.button('Ask') or query:
		if not query:
			st.info('Type a question above.')
		else:
			key = get_api_key()
			if not key:
				st.error('OpenRouter API key missing. Add it in the sidebar or set it in code.')
				st.stop()
			context_main = retrieve_context(query, vectorizer, matrix, pages, k=MAX_SNIPPETS)
			context_faq = []
			if faq_matrix is not None and faq_pages:
				context_faq = retrieve_context(query, vectorizer, faq_matrix, faq_pages, k=MAX_SNIPPETS)
			# Merge and keep top K overall
			context = sorted(context_main + context_faq, key=lambda s: s['score'], reverse=True)[:MAX_SNIPPETS]
			sys_p, user_p = build_prompt(query, context)
			try:
				answer = call_openrouter(key, MODEL_NAME, sys_p, user_p)
				st.subheader('Answer')
				st.write(answer)
				st.subheader('Context used')
				for s in context:
					label = 'Annual Report' if s.get('source') == 'annual_report' else 'FAQ PDF'
					st.markdown(f"**{label} · Page {s['page_number']}** · score {s['score']:.3f}")
					st.write(s['text'])
					st.divider()
				# Mark that a question has been answered
				st.session_state['question_answered'] = True
			except requests.HTTPError as e:
				st.error(f"OpenRouter API error: {e}\n{e.response.text if e.response is not None else ''}")
			except Exception as e:
				st.error(f"Unexpected error: {e}")

with col2:
	st.empty()

# --- Quick FAQs (moved below the search view) ---
# Only show Quick FAQs if no question has been answered yet
if not question_answered:
	st.markdown('### Quick FAQs')
	# Create columns for FAQ buttons using full width
	_faq_cols = st.columns([1, 1, 1, 1, 1, 1])  # 6 equal columns for 6 questions
	_faq_targets = TOP_FAQ_QUESTIONS[:6]
	for idx, q in enumerate(_faq_targets):
		with _faq_cols[idx]:
			if st.button(q, key=f"quick_faq_{idx}", use_container_width=True):
				# Use a different session state key to avoid conflict
				st.session_state['selected_question'] = q
				st.rerun()
else:
	# Show reset button when Quick FAQs are hidden
	st.markdown('### Quick FAQs')
	if st.button('🔄 Reset - Show Quick FAQs Again', key='reset_faqs', use_container_width=True):
		st.session_state['question_answered'] = False
		st.session_state['selected_question'] = ''
		st.rerun()

# --- Footer ---
st.markdown("""
<hr/>
<div style="text-align:center; opacity:0.7; font-size:14px;">
	<span>© 2025 Titan Company FAQ Chatbot · Built with Streamlit</span>
</div>
""", unsafe_allow_html=True)
