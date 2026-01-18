"""
Interactive PDF Question Answering - Ask questions directly in the terminal
"""

import sys
import os



# Add parent directory to path to allow importing from files module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


from files.Hybrid_pdf_qa_optimized import answer_pdf_question


def main():
    print("="*70)
    print("PDF Question Answering System")
    print("="*70)
    print()
    
    # Set PDF folder to data directory (contains all PDFs)
    # Use absolute path relative to the main project directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdfs_folder = os.path.join(parent_dir, "data")
    
    # Check if folder exists
    if not os.path.exists(pdfs_folder):
        print(f"\nError: Folder '{pdfs_folder}' does not exist!")
        sys.exit(1)
    
    # Count PDFs in the folder
    import glob
    pdf_files = glob.glob(os.path.join(pdfs_folder, "*.pdf"))
    pdf_count = len(pdf_files)
    print(f"Using folder: {pdfs_folder}")
    print(f"Found {pdf_count} PDF file(s) in the folder")
    print("\n" + "-"*70)
    print("You can now ask questions. Type 'exit' or 'quit' to stop.")
    print("-"*70)
    print()
    
    # Interactive loop
    while True:
        try:
            # Get question from user
            question = input("\nYour question: ").strip()
            
            # Check for exit commands
            if question.lower() in ['exit', 'quit', 'q', 'bye']:
                print("\nGoodbye!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            # Process and answer
            print("\nProcessing... This may take a moment...")
            print("-"*70)
            
            answer = answer_pdf_question(question, pdfs_folder)
            
            print("\nANSWER:")
            print("="*70)
            print(answer)
            print("="*70)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    main()
