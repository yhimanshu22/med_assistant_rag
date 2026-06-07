"""Sample medical context and query cases for fast pipeline/demo tests (no GPU required)."""

from langchain_core.documents import Document

# --- Sample document chunks (simulates ingested PDF content) ---

APLASTIC_ANEMIA_CHUNK = Document(
    page_content=(
        "Aplastic anemia is a rare, serious blood disorder in which the bone marrow "
        "fails to produce enough new blood cells. It can develop at any age and may "
        "occur suddenly or progress slowly. Common symptoms include fatigue, shortness "
        "of breath, frequent infections, and easy bruising or bleeding."
    ),
    metadata={"source": "aplastic_anemia_guide.pdf", "page": 12, "page_number": 13},
)

INFLUENZA_CHUNK = Document(
    page_content=(
        "Influenza (flu) is a contagious respiratory illness caused by influenza viruses. "
        "Symptoms include fever, cough, sore throat, runny or stuffy nose, muscle aches, "
        "headache, and fatigue. Annual vaccination is the primary preventive measure."
    ),
    metadata={"source": "influenza_factsheet.pdf", "page": 3, "page_number": 4},
)

MARROW_PANHYPOPLASIA_CHUNK = Document(
    page_content=(
        "Panhypoplasia of the marrow refers to a marked reduction in all three major "
        "blood cell lines — red cells, white cells, and platelets — due to suppressed "
        "bone marrow activity. It is a hallmark finding in severe aplastic anemia."
    ),
    metadata={"source": "hematology_reference.pdf", "page": 45, "page_number": 46},
)

DIABETES_CHUNK = Document(
    page_content=(
        "Type 2 diabetes mellitus is a chronic metabolic disorder characterized by "
        "insulin resistance and relative insulin deficiency. Management includes "
        "lifestyle modification, metformin, and other glucose-lowering agents."
    ),
    metadata={"source": "diabetes_management.pdf", "page": 8, "page_number": 9},
)

ALL_SAMPLE_CHUNKS = [
    APLASTIC_ANEMIA_CHUNK,
    INFLUENZA_CHUNK,
    MARROW_PANHYPOPLASIA_CHUNK,
    DIABETES_CHUNK,
]

# --- Query cases: question + expected grounding signals ---

SAMPLE_QUERY_CASES = [
    {
        "id": "aplastic_anemia_definition",
        "question": "What is aplastic anemia?",
        "primary_chunk": APLASTIC_ANEMIA_CHUNK,
        "expected_terms": ["bone marrow", "blood cells"],
        "dense_distance": 0.25,
    },
    {
        "id": "influenza_symptoms",
        "question": "What are the symptoms of Influenza?",
        "primary_chunk": INFLUENZA_CHUNK,
        "expected_terms": ["fever", "cough", "fatigue"],
        "dense_distance": 0.30,
    },
    {
        "id": "panhypoplasia_marrow",
        "question": "Explain panhypoplasia of the marrow",
        "primary_chunk": MARROW_PANHYPOPLASIA_CHUNK,
        "expected_terms": ["bone marrow", "blood cell"],
        "dense_distance": 0.28,
    },
]

WEAK_RETRIEVAL_CASE = {
    "id": "unrelated_cosmology",
    "question": "What is the mass of the sun in kilograms?",
    "primary_chunk": DIABETES_CHUNK,
    "dense_distance": 0.95,
}
