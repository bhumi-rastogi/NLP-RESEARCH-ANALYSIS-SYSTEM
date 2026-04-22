from setuptools import setup, find_packages

setup(
    name="nlp-research-analyzer",
    version="0.1.0",
    description="Milestone 1 — NLP Research Analysis System (no LLMs / agentic workflows)",
    author="Your Name",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "PyPDF2>=3.0.0",
        "nltk>=3.8",
        "scikit-learn>=1.3",
        "gensim>=4.3",
        "matplotlib>=3.7",
        "wordcloud>=1.9",
        "PyYAML>=6.0",
        "numpy>=1.24",
    ],
    entry_points={
        "console_scripts": [
            "nlp-analyze=nlp_research.__main__:main",
        ]
    },
)
