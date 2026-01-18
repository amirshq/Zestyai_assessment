"""
Experimentation Harness for PDF Question Answering
Tests multiple variations and provides comprehensive evaluation
"""

import re
import time
import json
from typing import List, Dict, Callable
from dataclasses import dataclass, asdict
import pandas as pd
from difflib import SequenceMatcher
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from files.pdf_question_answering import HybridPDFQuestionAnswerer


@dataclass
class ExperimentResult:
    """Store results from a single experiment run."""
    variation_name: str
    question: str
    expected_answer: str
    actual_answer: str
    execution_time: float
    token_estimate: int
    exact_match: bool
    similarity_score: float
    contains_key_info: bool


class PDFQAExperimentHarness:
    """Framework to evaluate multiple PDF QA variations."""
    
    def __init__(self):
        self.results: List[ExperimentResult] = []
        self.variations: Dict[str, Callable] = {}
    
    def register_variation(self, name: str, variation_func: Callable):
        """Register a variation for testing."""
        self.variations[name] = variation_func
        print(f"Registered variation: {name}")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def check_key_information(self, answer: str, expected: str) -> bool:
        """Check if answer contains key information from expected answer."""
        expected_lower = expected.lower()
        answer_lower = answer.lower()
        
        # For list-type answers
        if any(marker in expected_lower for marker in ['list', '•', '-', '1.', '2.']):
            expected_items = re.findall(r'(?:^|\n)\s*[•\-\*\d\.]\s*([^\n]+)', expected)
            if expected_items:
                matches = sum(1 for item in expected_items if item.lower().strip() in answer_lower)
                return matches >= len(expected_items) * 0.7
        
        # For numerical answers
        expected_numbers = set(re.findall(r'\d+', expected))
        if expected_numbers:
            answer_numbers = set(re.findall(r'\d+', answer))
            if len(expected_numbers.intersection(answer_numbers)) / len(expected_numbers) >= 0.8:
                return True
        
        # For general answers - check key word overlap
        expected_words = set(expected_lower.split())
        answer_words = set(answer_lower.split())
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        key_words = expected_words - common_words
        
        if key_words:
            return len(key_words.intersection(answer_words)) / len(key_words) >= 0.6
        return False
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def run_experiment(self, variation_name: str, question: str, 
                      pdfs_folder: str, expected_answer: str = "") -> ExperimentResult:
        """Run a single experiment with one variation."""
        if variation_name not in self.variations:
            raise ValueError(f"Variation '{variation_name}' not registered")
        
        print(f"\n{'='*80}\nRunning: {variation_name}\nQuestion: {question}\n{'='*80}")
        
        start_time = time.time()
        try:
            actual_answer = self.variations[variation_name](question, pdfs_folder)
        except Exception as e:
            actual_answer = f"ERROR: {str(e)}"
        execution_time = time.time() - start_time
        
        # Calculate metrics
        exact_match = (actual_answer.strip().lower() == expected_answer.strip().lower()) if expected_answer else False
        similarity_score = self.calculate_similarity(actual_answer, expected_answer) if expected_answer else 0.0
        contains_key_info = self.check_key_information(actual_answer, expected_answer) if expected_answer else False
        token_estimate = self.estimate_tokens(actual_answer)
        
        result = ExperimentResult(
            variation_name=variation_name,
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            execution_time=execution_time,
            token_estimate=token_estimate,
            exact_match=exact_match,
            similarity_score=similarity_score,
            contains_key_info=contains_key_info
        )
        
        self.results.append(result)
        print(f"✓ Completed in {execution_time:.2f}s | Similarity: {similarity_score:.2%} | Key info: {contains_key_info}")
        print(f"  Answer: {actual_answer[:200]}...")
        
        return result
    
    def run_all_experiments(self, test_cases: List[Dict]):
        """Run all variations on all test cases."""
        print(f"\n{'='*80}\nSTARTING EXPERIMENTATION HARNESS\n{'='*80}")
        print(f"Variations: {len(self.variations)} | Test cases: {len(test_cases)} | Total: {len(self.variations) * len(test_cases)}")
        
        for variation_name in self.variations.keys():
            for test_case in test_cases:
                self.run_experiment(
                    variation_name=variation_name,
                    question=test_case['question'],
                    pdfs_folder=test_case['pdfs_folder'],
                    expected_answer=test_case.get('expected_answer', '')
                )
    
    def generate_report(self) -> pd.DataFrame:
        """Generate comparative report of all experiments."""
        if not self.results:
            print("No results to report")
            return pd.DataFrame()
        
        df = pd.DataFrame([asdict(r) for r in self.results])
        summary = df.groupby('variation_name').agg({
            'exact_match': 'sum',
            'similarity_score': 'mean',
            'contains_key_info': 'sum',
            'execution_time': 'mean',
            'token_estimate': 'mean'
        }).round(3)
        
        summary.columns = ['Exact Matches', 'Avg Similarity', 'Key Info Matches', 'Avg Time (s)', 'Avg Tokens']
        return summary
    
    def print_detailed_report(self):
        """Print detailed comparison report."""
        if not self.results:
            print("No results to report")
            return
        
        print(f"\n\n{'='*80}\nEXPERIMENT RESULTS SUMMARY\n{'='*80}")
        print("\n" + self.generate_report().to_string())
        
        print(f"\n\n{'='*80}\nDETAILED RESULTS BY QUESTION\n{'='*80}")
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        for question in df['question'].unique():
            print(f"\n\nQuestion: {question}\n{'-'*80}")
            question_results = df[df['question'] == question]
            
            for _, row in question_results.iterrows():
                print(f"\nVariation: {row['variation_name']}")
                print(f"  Time: {row['execution_time']:.2f}s | Similarity: {row['similarity_score']:.2%} | Key info: {row['contains_key_info']}")
                print(f"  Answer: {row['actual_answer'][:300]}...")
                if row['expected_answer']:
                    print(f"  Expected: {row['expected_answer'][:300]}...")
    
    def export_results(self, filepath: str = "experiment_results.json"):
        """Export results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"\nResults exported to {filepath}")


# Variation Implementations
def variation_1_baseline(question: str, pdfs_folder: str) -> str:
    """Baseline: Standard settings with small embedding model."""
    qa = HybridPDFQuestionAnswerer(model="gpt-4o-mini", embedding_model="text-embedding-3-small")
    chunks = qa.process_pdfs_folder(pdfs_folder)
    relevant_chunks, _ = qa.get_relevant_chunks(question, chunks)
    return qa.answer_question(question, relevant_chunks)


def variation_2_larger_embedding(question: str, pdfs_folder: str) -> str:
    """Large embedding model for better semantic understanding."""
    qa = HybridPDFQuestionAnswerer(model="gpt-4o-mini", embedding_model="text-embedding-3-large")
    chunks = qa.process_pdfs_folder(pdfs_folder)
    relevant_chunks, _ = qa.get_relevant_chunks(question, chunks)
    return qa.answer_question(question, relevant_chunks)


def variation_3_smaller_chunks(question: str, pdfs_folder: str) -> str:
    """Smaller chunks (1500) for more granular retrieval."""
    qa = HybridPDFQuestionAnswerer(model="gpt-4o-mini", embedding_model="text-embedding-3-small", 
                                   chunk_size=1500, chunk_overlap=150)
    chunks = qa.process_pdfs_folder(pdfs_folder)
    relevant_chunks, _ = qa.get_relevant_chunks(question, chunks)
    return qa.answer_question(question, relevant_chunks)


def variation_4_larger_chunks(question: str, pdfs_folder: str) -> str:
    """Larger chunks (3000) for more context per chunk."""
    qa = HybridPDFQuestionAnswerer(model="gpt-4o-mini", embedding_model="text-embedding-3-small",
                                   chunk_size=3000, chunk_overlap=300)
    chunks = qa.process_pdfs_folder(pdfs_folder)
    relevant_chunks, _ = qa.get_relevant_chunks(question, chunks)
    return qa.answer_question(question, relevant_chunks)


if __name__ == "__main__":
    harness = PDFQAExperimentHarness()
    
    # Register variations
    harness.register_variation("V1_Baseline_SmallEmbedding", variation_1_baseline)
    harness.register_variation("V2_LargeEmbedding", variation_2_larger_embedding)
    harness.register_variation("V3_SmallerChunks", variation_3_smaller_chunks)
    harness.register_variation("V4_LargerChunks", variation_4_larger_chunks)
    
    # Define test cases
    data_folder = str(Path(__file__).parent.parent / "data")
    test_cases = [
        {
            'question': "List all rating plan rules",
            'pdfs_folder': data_folder,
            'expected_answer': "Limits of Liability and Coverage Relationships, Rating Perils, Base Rates, Policy Type Factor, Policy Tier Guidelines, Amount of Insurance / Deductibles, Hurricane Deductibles, Windstorm / Hail Deductibles, Policy Territory Determination, Distance to Coast Factor, Public Protection Class Factors, Age of Home Factor, Year Built Factor, Account Discount, Roof Type Factor, Dwelling Usage Factor, Increased Limits, Protective Device Discount, Affinity Discount, Association Discount, Oil Tank Factor, Pool Factor, Trampoline Factor, Roof Condition Factor, Tree Overhang Factor, Solar Panel Factor, Secondary Heating Source Factor, Windstorm Mitigation Discounts, Endorsement Combination Discount, Loss History Rating, Claims Free Discount, Underwriting Experience, Minimum Premium"
        },
        {
            'question': "Using the Base Rate and the applicable Mandatory Hurricane Deductible Factor, calculate the unadjusted Hurricane premium for an HO3 policy with a $750,000 Coverage A limit located 3,000 feet from the coast in a Coastline Neighborhood.",
            'pdfs_folder': data_folder,
            'expected_answer': "$604"
        },
        {
            'question': "What is the main topic of the document?",
            'pdfs_folder': data_folder,
            'expected_answer': ""
        }
    ]
    
    # Run experiments
    harness.run_all_experiments(test_cases)
    harness.print_detailed_report()
    harness.export_results("pdf_qa_experiment_results.json")
    
    # Print key findings
    print(f"\n\n{'='*80}\nKEY FINDINGS\n{'='*80}")
    print("""
Chunking: Smaller (1500) = more precise | Larger (3000) = more context
Embedding: Small model = faster/cheaper | Large model = better semantic understanding
Next Steps: Adaptive chunking, query expansion, re-ranking, multi-hop reasoning
    """)