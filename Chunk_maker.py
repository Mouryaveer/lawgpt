import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any
import tiktoken

class LegalDocumentChunker:
    """
    Chunks legal documents from India Code JSON files with proper metadata attachment.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        model: str = "cl100k_base"
    ):
        """
        Initialize the chunker with configurable parameters.
        
        Args:
            chunk_size: Target size for each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            model: Tokenizer model to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(model)
        self.total_chunks = 0
        self.stats = {}
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.encoding.encode(text))
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text while preserving semantic breaks.
        
        Args:
            text: Raw text to normalize
            
        Returns:
            Normalized text
        """
        # Replace excessive newlines with double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Replace multiple spaces with single space (except newlines)
        text = re.sub(r'[ ]{2,}', ' ', text)
        
        # Trim whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using token-based approach.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) < 30:
            return []
        
        # Normalize the text first
        text = self.normalize_text(text)
        
        # Encode the entire text
        tokens = self.encoding.encode(text)
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Get chunk of tokens
            end_idx = start_idx + self.chunk_size
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Only add non-empty chunks
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # Move to next chunk with overlap
            start_idx += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def create_chunk_id(
        self,
        year: str,
        act_index: int,
        section_index: int,
        chunk_index: int
    ) -> str:
        """
        Create a unique identifier for a chunk.
        
        Format: <year>_<act_index>_<section_index>_<chunk_index>
        """
        return f"{year}_{act_index}_{section_index}_{chunk_index}"
    
    def process_section(
        self,
        section: Dict[str, Any],
        year: str,
        act_index: int,
        section_index: int,
        base_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process a single section and create chunks with metadata.
        
        Args:
            section: Section data containing text and metadata
            year: Year of the act
            act_index: Index of the act in the year
            section_index: Index of the section in the act
            base_metadata: Base metadata from act level
            
        Returns:
            List of chunk objects with metadata
        """
        section_text = section.get("text", "")
        
        # Skip if text is too short
        if len(section_text.strip()) < 30:
            return []
        
        # Get section-specific metadata
        section_title = section.get("title", "Full Act PDF")
        section_url = section.get("url", "")
        
        # Create chunks
        text_chunks = self.chunk_text(section_text)
        
        if not text_chunks:
            return []
        
        total_chunks = len(text_chunks)
        chunk_objects = []
        
        for chunk_idx, chunk_text in enumerate(text_chunks, start=1):
            chunk_id = self.create_chunk_id(
                year, act_index, section_index, chunk_idx
            )
            
            metadata = {
                "year": year,
                "act_title": base_metadata.get("act_title", ""),
                "section_title": section_title,
                "source": section.get("source", "pdf"),
                "chunk_index": chunk_idx,
                "total_chunks_in_section": total_chunks,
                "year_url": base_metadata.get("year_url", ""),
                "act_url": base_metadata.get("act_url", ""),
                "section_url": section_url,
                "pdf_url": base_metadata.get("pdf_url", "")
            }
            
            chunk_obj = {
                "id": chunk_id,
                "text": chunk_text,
                "metadata": metadata
            }
            
            chunk_objects.append(chunk_obj)
            self.total_chunks += 1
        
        return chunk_objects
    
    def process_act(
        self,
        act: Dict[str, Any],
        year: str,
        year_url: str,
        act_index: int
    ) -> List[Dict[str, Any]]:
        """
        Process a single act and all its sections.
        
        Args:
            act: Act data containing sections and metadata
            year: Year of the act
            year_url: URL for the year
            act_index: Index of the act in the year
            
        Returns:
            List of all chunk objects from all sections
        """
        act_title = act.get("title", "Untitled Act")
        act_url = act.get("url", "")
        pdf_url = act.get("pdf_url", "")
        
        base_metadata = {
            "year_url": year_url,
            "act_title": act_title,
            "act_url": act_url,
            "pdf_url": pdf_url
        }
        
        all_chunks = []
        sections = act.get("sections", [])
        
        for section_index, section in enumerate(sections, start=1):
            section_chunks = self.process_section(
                section,
                year,
                act_index,
                section_index,
                base_metadata
            )
            all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def process_year_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single year JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of all chunk objects from the file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        year = data.get("year", "Unknown")
        year_url = data.get("url", "")
        acts = data.get("acts", [])
        
        all_chunks = []
        
        for act_index, act in enumerate(acts, start=1):
            act_chunks = self.process_act(act, year, year_url, act_index)
            all_chunks.extend(act_chunks)
        
        # Update stats
        self.stats[year] = {
            "total_acts": len(acts),
            "total_chunks": len(all_chunks)
        }
        
        print(f"Year: {year} | Acts: {len(acts)} | Total Chunks: {len(all_chunks)}")
        
        return all_chunks
    
    def process_directory(
        self,
        input_dir: str,
        output_file: str = "chunked_documents.jsonl"
    ):
        """
        Process all JSON files in a directory and write chunks to output.
        
        Args:
            input_dir: Directory containing year JSON files
            output_file: Output JSONL file path
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        # Get all JSON files
        json_files = sorted(input_path.glob("*.json"))
        
        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return
        
        print(f"Found {len(json_files)} JSON files to process\n")
        print("=" * 60)
        
        # Process each file and write chunks
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for json_file in json_files:
                print(f"\nProcessing: {json_file.name}")
                
                try:
                    chunks = self.process_year_file(str(json_file))
                    
                    # Write each chunk as a JSON line
                    for chunk in chunks:
                        out_f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                    
                except Exception as e:
                    print(f"Error processing {json_file.name}: {str(e)}")
                    continue
        
        print("\n" + "=" * 60)
        print(f"\n✅ Processing complete!")
        print(f"Total chunks created: {self.total_chunks}")
        print(f"Output written to: {output_file}")
        
        # Print summary stats
        print("\n📊 Summary by Year:")
        for year in sorted(self.stats.keys()):
            stats = self.stats[year]
            print(f"  {year}: {stats['total_acts']} acts, {stats['total_chunks']} chunks")


def main():
    """
    Main execution function.
    """
    # Configuration
    INPUT_DIRECTORY = "scraped_acts"  # Directory with year_XXXX.json files
    OUTPUT_FILE = "chunked_documents.jsonl"
    
    # Initialize chunker with parameters from the plan
    chunker = LegalDocumentChunker(
        chunk_size=1000,      # 1000-1200 tokens
        chunk_overlap=150     # 150-200 tokens
    )
    
    # Process all files
    chunker.process_directory(INPUT_DIRECTORY, OUTPUT_FILE)


if __name__ == "__main__":
    main()