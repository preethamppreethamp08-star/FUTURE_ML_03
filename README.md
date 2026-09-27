Resume / Candidate Screening System

An explainable ML system that reads resumes, compares them against a job description, ranks candidates by role fit, and reports missing skills — built for Future Interns ML Task 3.

Problem

Hiring teams receive hundreds of resumes per role and manually screening them is slow and inconsistent. This project automates the first pass: scoring, ranking, and explaining candidate fit against a job description.

Dataset

Uses the Kaggle Resume Dataset (2,484 real resumes across 24 job categories, scraped from livecareer.com). Columns used: ID, Resume_str, Category.

How It Works
Text cleaning — lowercases resume text, strips punctuation/noise from the scraped source, collapses whitespace.
Skill extraction (skills_dictionary.py) — a curated dictionary of ~28 canonical skills (Python, SQL, TensorFlow, NLP, Docker, etc.), each with alias spellings. This is a transparent keyword match rather than a black-box model, so every match can be traced back to the exact word in the resume.
Similarity scoring — resumes and the job description are vectorized with TF-IDF (unigrams + bigrams) and compared with cosine similarity. TF-IDF down-weights common words and up-weights distinctive terms, so a resume that echoes the JD's specific vocabulary scores higher than one sharing only generic words.
Final ranking score = 0.6 × similarity_score + 0.4 × skill_match_pct
Similarity score: overall text closeness to the JD (0–100).
Skill match %: fraction of the JD's required skills found literally in the resume.
This blend rewards holistic fit while still crediting candidates who name the exact tools asked for.
Skill gap identification — for each candidate: required_skills − matched_skills = the missing-skills list shown in the report, so a recruiter immediately sees what's absent and why a candidate ranked where they did.
Why This Is Explainable

Every number traces to something concrete — you can point to the exact skill keyword matched, and the similarity score comes from term overlap, not a hidden embedding. That satisfies the brief's requirement to explain how resumes are scored and why one candidate outranks another to a non-technical audience (a recruiter or HR manager).

Project Structure
resume_screening/
├── resume_screening.py     # main pipeline — run this
├── skills_dictionary.py    # skill list + extraction logic
├── data/
│   ├── resumes.csv          # Kaggle Resume Dataset (ID, Resume_str, Category)
│   └── job_description.txt  # target role description (edit this per role)
└── results/
    ├── ranked_candidates.csv
    └── ranking_chart.png
Running It
bash
pip install pandas scikit-learn matplotlib
python resume_screening.py --category "Information-Technology"

--category filters the resume pool to one Kaggle category so you're screening a relevant group. Available categories: HR, Designer, Information-Technology, Teacher, Advocate, Business-Development, Healthcare, Fitness, Agriculture, BPO, Sales, Consultant, Digital-Media, Automobile, Chef, Finance, Apparel, Engineering, Accountant, Construction, Public-Relations, Banking, Arts, Aviation. Omit --category to screen the full dataset.

--top N controls how many ranked candidates are printed/plotted (default 15).

Sample Output
Rank 1: 10624813  (Final score: 35.8/100)
   Text similarity to job description: 26.4/100
   Required-skill coverage: 50.0%  -> matched: java, machine learning, python, sql
   Missing skills: docker, git, rest api, scikit-learn
Known Limitation

Keyword matching doesn't understand negation — a resume saying "no experience with machine learning" still registers as a match on "machine learning" because the phrase is present. A natural next step is a negation check: skip a match if a word like "no," "not," or "without" appears a few tokens before it.

Ideas for Extending
Negation handling (see above).
Weight some JD skills as "must-have" vs. "nice-to-have" for a smarter skill_match_pct.
Screen against multiple job descriptions to compare how the same resume pool ranks for different roles.
Swap the keyword dictionary for a spaCy PhraseMatcher for fuzzier matching (e.g., catching typos or reworded skill mentions).
Built For

Future Interns — Machine Learning Task 3 (Resume / Candidate Screening System), 2026.
