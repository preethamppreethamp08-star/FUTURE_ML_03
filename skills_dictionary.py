"""
Curated skill dictionary for the resume screening system.

Each entry maps a canonical skill name to a list of surface forms / aliases
that might appear in resume or job description text. This keyword-based
approach is transparent and explainable (a key requirement of the task),
unlike a black-box embedding model.
"""

SKILL_DICTIONARY = {
    "python": ["python"],
    "sql": ["sql", "mysql", "postgresql", "postgres"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "tensorflow": ["tensorflow"],
    "keras": ["keras"],
    "pytorch": ["pytorch", "torch"],
    "nlp": ["nlp", "natural language processing"],
    "spacy": ["spacy"],
    "nltk": ["nltk"],
    "power bi": ["power bi", "powerbi"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "data visualization": ["data visualization", "visualisation"],
    "deep learning": ["deep learning"],
    "machine learning": ["machine learning", "ml "],
    "java": [" java "],
    "c++": ["c++"],
    "javascript": ["javascript", "js "],
    "react": ["react"],
    "node.js": ["node.js", "nodejs", "node js"],
    "docker": ["docker"],
    "git": ["git "],
    "rest api": ["rest api", "rest apis", "restful api"],
    "excel": ["excel"],
    "statistics": ["statistics", "statistical"],
    "text classification": ["text classification"],
    "text similarity": ["text similarity", "similarity scoring"],
}


def extract_skills(text: str) -> set:
    """Return the set of canonical skills found in a block of text."""
    text_lower = f" {text.lower()} ".replace("\n", " ").replace(",", " ").replace(".", " ")
    found = set()
    for canonical, aliases in SKILL_DICTIONARY.items():
        for alias in aliases:
            if alias in text_lower:
                found.add(canonical)
                break
    return found
