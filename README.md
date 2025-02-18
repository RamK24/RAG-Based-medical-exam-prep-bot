### Overview

The bot is based on this <a href='https://huggingface.co/datasets/MedRAG/textbooks'> Dataset </a> from huggingface. 
Textbooks is a collection of 18 widely used medical textbooks, which are important references for students taking the United States Medical Licensing Examination (USLME).
<ul>
    <li>Anatomy - Gray</li>
    <li>Biochemistry - Lippincott</li>
    <li>Cell Biology - Alberts</li>
    <li>First Aid - Step 1</li>
    <li>First Aid - Step 2</li>
    <li>Gynecology - Novak</li>
    <li>Histology - Ross</li>
    <li>Immunology - Janeway</li>
    <li>Internal Medicine - Harrison</li>
    <li>Neurology - Adams</li>
    <li>Obstetrics - Williams</li>
    <li>Pathology - Robbins</li>
    <li>Pathoma - Husain</li>
    <li>Pediatrics - Nelson</li>
    <li>Pharmacology - Katzung</li>
    <li>Physiology - Levy</li>
    <li>Psychiatry - DSM-5</li>
    <li>Surgery - Schwartz</li>
</ul>

### Usage
```sh
git clone https://github.com/RamK24/RAG-Based-medical-exam-prep-bot.git
```

```sh
python -m venv venv_name
source  venv_name/bin/activate
pip install -r requirements.txt
```

Webapp
```sh
python main.py
```
click on the local host link to use the webapp

```text
input = 'A 45-year-old man with chronic alcohol use presents with confusion, ataxia, and ophthalmoplegia. What is the most likely diagnosis?
A) Wernicke encephalopathy
B) Korsakoff syndrome
C) Hepatic encephalopathy
D) Multiple sclerosis'
output - Likely answer: Wernicke encephalopathy
```

