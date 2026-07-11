"""PDF utilities for downloading and extracting text optimized for vector embeddings."""
import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


# Public API - maintain exact same function signatures as original
__all__ = ['extract_text_from_pdf', 'clean_pdf_text']


def clean_pdf_text(text: str) -> str:
    """
    Clean extracted PDF text optimized for vector embeddings:
    - Preserves semantic meaning and context
    - Removes noise (headers, footers, page numbers)
    - Fixes encoding and formatting issues
    - Creates coherent text blocks for better embeddings
    - Maintains logical document structure
    """
    if not text or not text.strip():
        return ""
    
    # Fix encoding issues first
    text = _fix_encoding(text)
    
    # Fix reading order issues
    text = _fix_reading_order(text)
    
    # Remove headers/footers and page numbers
    lines = text.split('\n')
    lines = _remove_headers_footers(lines)
    
    # Fix hyphenation and broken words
    text = '\n'.join(lines)
    text = _fix_hyphenation(text)
    text = _fix_broken_words(text)
    
    # Process lines with structure preservation
    lines = text.split('\n')
    cleaned_lines = _preserve_structure(lines)
    
    # Join into coherent paragraphs for better embeddings
    text = '\n'.join(cleaned_lines)
    
    # Fix spacing issues
    text = _fix_spacing(text)
    
    # Remove excessive whitespace but preserve paragraph breaks
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{4,}', '\n\n', text)  # Max 2 newlines (one blank line)
    
    # Remove lines that are just punctuation or single characters
    lines = text.split('\n')
    lines = [l for l in lines if l.strip() and (len(l.strip()) > 1 or l.strip().isalnum())]
    text = '\n'.join(lines)
    
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and clean text from PDF using pdfminer.six, optimized for embeddings"""
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
        
        # Optimized LAParams for better text coherence
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
            detect_vertical=False,
            all_texts=True,
        )
        
        text = extract_text(pdf_path, laparams=laparams)
        
        if not text:
            logger.warning(f"No text extracted from PDF: {pdf_path}")
            return ""
        
        # Clean the extracted text
        text = clean_pdf_text(text)
        
        logger.info(f"Extracted and cleaned {len(text)} characters from {pdf_path}")
        return text
    except ImportError as e:
        logger.error(f"pdfminer.six not installed: {e}")
        return ""
    except Exception as e:
        logger.error(f"PDF extraction error for {pdf_path}: {e}")
        return ""


# Helper functions for advanced cleaning

def _fix_encoding(text: str) -> str:
    """Fix common PDF encoding issues."""
    replacements = {
        '\u2019': "'", '\u2018': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-',
        '\u2026': '...', '\xa0': ' ',
        '\u00a0': ' ', '\ufeff': '',
        '\u2022': '-', '\u2023': '-',
        '\u25cf': '-', '\u25e6': '-',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove problematic Unicode control characters
    text = ''.join(char if ord(char) < 0x10000 and ord(char) >= 32 or char in '\n\r\t' else ' ' for char in text)
    return text


def _fix_reading_order(text: str) -> str:
    """
    Fix common reading order issues in PDFs where title, date, or 
    preamble text appears in wrong position.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    if not lines:
        return text
    
    # Detect document structure elements
    title_idx = None
    act_number_idx = None
    date_idx = None
    first_section_idx = None
    
    for i, line in enumerate(lines[:50]):  # Only check first 50 lines
        # Detect title (all caps, contains "ACT")
        if re.match(r'^[A-Z\s,\-]+ACT[A-Z\s,\-]*\d{4}', line):
            title_idx = i
        
        # Detect act number
        if re.match(r'^ACT NO\.\s+\d+\s+OF\s+\d{4}', line):
            act_number_idx = i
        
        # Detect date
        if re.match(r'^\[.*?\d{4}\.?\]$', line):
            date_idx = i
        
        # Detect first section/clause
        if re.match(r'^(?:I+\.|1\.|Section\s+I+|SECTION\s+1)', line, re.IGNORECASE) and first_section_idx is None:
            first_section_idx = i
    
    # Reorder if title/date appear after content starts
    if first_section_idx is not None and any([
        title_idx and title_idx > first_section_idx,
        act_number_idx and act_number_idx > first_section_idx,
        date_idx and date_idx > first_section_idx
    ]):
        # Extract header elements
        header_parts = []
        content_parts = []
        
        for i, line in enumerate(lines):
            if i == title_idx or i == act_number_idx or i == date_idx:
                header_parts.append((i, line))
            else:
                content_parts.append(line)
        
        # Sort header parts by their typical order
        header_parts.sort(key=lambda x: (
            0 if 'ACT' in x[1] and 'NO.' not in x[1] else
            1 if 'ACT NO.' in x[1] else
            2  # date
        ))
        
        # Reconstruct with proper order
        reordered = [hp[1] for hp in header_parts] + [''] + content_parts
        text = '\n'.join(reordered)
    
    return text


def _remove_headers_footers(lines: List[str], threshold: int = 3) -> List[str]:
    """
    Remove repeated headers/footers that appear on multiple pages.
    Critical for embeddings to avoid duplicate content.
    """
    if len(lines) < 10:
        return lines
    
    line_counts = Counter()
    
    # Count occurrences of non-empty lines
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) > 2:
            line_counts[stripped] += 1
    
    # Identify repeated lines (headers/footers)
    page_num_pattern = re.compile(r'^\d+$')
    all_caps_short_pattern = re.compile(r'^[A-Z\s\d,\.;:]+$')
    
    repeated = set()
    for line, count in line_counts.items():
        # Remove if repeated frequently and matches header/footer patterns
        if count >= threshold:
            is_page_num = page_num_pattern.match(line)
            is_short_caps = all_caps_short_pattern.match(line) and len(line) < 50
            is_very_short = len(line) < 20
            
            if is_page_num or (is_short_caps and count >= 5) or (is_very_short and count >= 8):
                repeated.add(line)
    
    # Filter out repeated lines
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped not in repeated:
            result.append(line)
    
    return result


def _fix_hyphenation(text: str) -> str:
    """Fix words hyphenated across line breaks."""
    # Standard hyphenation: word- \n word
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    return text


def _fix_broken_words(text: str) -> str:
    """
    Fix words broken across lines without hyphens.
    Conservative approach to avoid false joins.
    """
    def maybe_join(match):
        word1, word2 = match.group(1), match.group(2)
        
        # Don't join if second word starts with capital (likely new sentence)
        if word2[0].isupper():
            return f"{word1}\n{word2}"
        
        # Don't join common complete words
        complete = {
            'the', 'and', 'for', 'with', 'from', 'that', 'this', 'have', 
            'which', 'their', 'would', 'there', 'been', 'shall', 'will', 
            'such', 'said', 'made', 'may', 'any', 'all', 'not', 'but',
            'are', 'was', 'were', 'has', 'had', 'can', 'who', 'when'
        }
        if word1.lower() in complete or word2.lower() in complete:
            return f"{word1} {word2}"
        
        # Check if joining creates suspicious patterns
        joined = word1 + word2
        suspicious = ['toto', 'inin', 'ofof', 'thethe', 'eded', 'anan', 'isis']
        if any(sus in joined.lower() for sus in suspicious):
            return f"{word1} {word2}"
        
        # If word1 ends and word2 starts with common morphemes, likely separate
        if word1.lower().endswith(('ing', 'ion', 'ed', 'ly', 'er', 'est', 'tion', 'ness', 'ment')):
            return f"{word1} {word2}"
        
        # Join if it looks like a broken word
        return joined
    
    # Match: lowercase ending \n lowercase starting
    text = re.sub(r'([a-z]{3,})\s*\n\s*([a-z]{3,})', maybe_join, text)
    
    return text


def _fix_spacing(text: str) -> str:
    """Fix spacing issues while preserving intentional breaks."""
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Ensure space after punctuation (if followed by letter)
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
    
    # Fix common spacing issues with parentheses
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    return text


def _preserve_structure(lines: List[str]) -> List[str]:
    """
    Preserve document structure for better embeddings.
    Joins related text while keeping sections separate.
    """
    # Patterns for legal/structured documents
    section_pattern = re.compile(
        r'^(?:'
        r'(?:SECTION|Section|Rule|Article|Chapter|Part|Schedule|Provision|Clause)\s+[A-Z0-9]+|'
        r'[IVX]+\.\s|'  # Roman numerals
        r'[A-Z]\.\s|'   # Single letter sections
        r'\d+\.\s+[A-Z]|'  # Numbered sections starting with capital
        r'\([a-z0-9]+\)\s*[A-Z]|'  # Parenthetical sections
        r'Provided|Whereas|Enacted'  # Legal keywords
        r')',
        re.IGNORECASE
    )
    
    list_pattern = re.compile(r'^[\-•·∙◦▪▫]\s+|\(\d+\)|\([a-z]\)\s+|\d+\)\s+')
    
    # All caps short lines (likely headings)
    heading_pattern = re.compile(r'^[A-Z][A-Z\s\d,\.;:\-]{8,80}$')
    
    cleaned = []
    prev_was_break = True  # Start with break
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip truly empty lines but track them
        if not stripped:
            if cleaned and not prev_was_break:
                cleaned.append('')
                prev_was_break = True
            continue
        
        # Skip very short lines (likely artifacts) unless structural
        if len(stripped) < 3 and not re.match(r'^[IVX]+\.$', stripped):
            continue
        
        # Detect structural elements
        is_section = bool(section_pattern.match(stripped))
        is_heading = bool(heading_pattern.match(stripped)) and not is_section
        is_list = bool(list_pattern.match(stripped))
        
        # Tables: lines with multiple spaces (columns)
        is_table = len(re.findall(r'\s{3,}', stripped)) >= 2
        
        # Structural elements get breaks before them
        if is_section or is_heading:
            if cleaned and cleaned[-1] != '':
                cleaned.append('')
            cleaned.append(stripped)
            prev_was_break = True
            continue
        
        # Tables and lists: preserve as-is
        if is_table or is_list:
            cleaned.append(stripped)
            prev_was_break = False
            continue
        
        # Regular text: join with previous if appropriate
        should_join = (
            cleaned and
            cleaned[-1] and
            not prev_was_break and
            _should_join_with_previous(stripped, cleaned[-1])
        )
        
        if should_join:
            # Join with previous line (create paragraph)
            cleaned[-1] = cleaned[-1] + ' ' + stripped
        else:
            cleaned.append(stripped)
        
        prev_was_break = False
    
    return cleaned


def _should_join_with_previous(current: str, previous: str) -> bool:
    """
    Determine if current line should join with previous line.
    Optimized for creating coherent text blocks for embeddings.
    """
    if not previous or not current:
        return False
    
    # Don't join if previous line ends with strong punctuation
    if previous.rstrip()[-1:] in '.!?:':
        return False
    
    # Don't join if current starts with capital (likely new sentence)
    if current[0].isupper() and not previous.endswith(','):
        return False
    
    # Don't join if current is a list item or section
    if re.match(r'^[\-•·∙◦▪▫\d\(]', current):
        return False
    
    # Don't join if previous is very short (likely heading)
    if len(previous) < 40 and previous[0].isupper():
        return False
    
    # Join if previous doesn't end with sentence-ending punctuation
    # and current doesn't start with structural markers
    return True