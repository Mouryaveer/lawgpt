"""IndiaCode PDF-only scraper using Playwright."""
import asyncio
import json
import os
import logging
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from urllib.parse import urljoin

from utils.pdf_utils import extract_text_from_pdf
from utils.save_utils import safe_json_write, load_year_data, get_scraped_act_urls

# Constants
OUTPUT_DIR = "scraped_acts"
TEMP_DIR = "temp_pdfs"
TEMP_YEARS_DIR = "temp_years"
LOG_DIR = "logs"

# Setup logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'run.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Fix Windows console encoding
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Selectors dictionary
SELECTORS = {
    # Act listing table
    "act_links": "div.panel.panel-primary table tbody tr td a",
    
    # Act page
    "pdf_link": "a[href*='.pdf']",
}


async def retry_with_backoff(func, max_retries=3, base_delay=1):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        
    Returns:
        Result of the function, or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"All retries failed: {e}")
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
            await asyncio.sleep(delay)
    return None


async def scrape_act_pdf(context, act_url: str):
    """
    Scrape act content via PDF extraction only.
    
    Args:
        context: Playwright browser context
        act_url: URL of the act to scrape
        
    Returns:
        Dictionary with act data including PDF text
    """
    page = None
    try:
        page = await context.new_page()
        
        # Navigate to act page
        async def navigate_to_act():
            await page.goto(act_url, wait_until='domcontentloaded', timeout=10000)
        await retry_with_backoff(navigate_to_act)
        
        # Get act title
        title = await page.title()
        if not title or title.strip() == "":
            title = "Unknown Act"
        logger.info(f"Processing: {title[:60]}...")
        
        # Look for PDF link
        pdf_link = await page.query_selector(SELECTORS["pdf_link"])
        if not pdf_link:
            logger.warning(f"  No PDF link found")
            await page.close()
            return {
                "title": title,
                "url": act_url,
                "source": "no_pdf",
                "pdf_url": None,
                "sections": []
            }
        
        pdf_href = await pdf_link.get_attribute("href")
        if not pdf_href:
            logger.warning("  PDF link has no href")
            await page.close()
            return {
                "title": title,
                "url": act_url,
                "source": "no_pdf",
                "pdf_url": None,
                "sections": []
            }
        
        pdf_url = urljoin(act_url, pdf_href)
        logger.info(f"  → Downloading PDF: {pdf_url}")
        
        # Use JavaScript fetch for reliable PDF download with session cookies
        try:
            pdf_bytes = await page.evaluate("""
                async (url) => {
                    const response = await fetch(url, {credentials: 'include'});
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    const buffer = await response.arrayBuffer();
                    return Array.from(new Uint8Array(buffer));
                }
            """, pdf_url)
            
            # Save to temp file
            os.makedirs(TEMP_DIR, exist_ok=True)
            temp_pdf = os.path.join(TEMP_DIR, f"{hash(act_url)}.pdf")
            
            with open(temp_pdf, 'wb') as f:
                f.write(bytes(pdf_bytes))
            
            logger.info(f"  Downloaded {len(pdf_bytes)} bytes")
            
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(temp_pdf)
            
            # Clean up temp file
            try:
                os.remove(temp_pdf)
            except:
                pass
            
            await page.close()
            
            if pdf_text and pdf_text.strip():
                word_count = len(pdf_text.split())
                logger.info(f"  ✓ Extracted {word_count} words from PDF")
                return {
                    "title": title,
                    "url": act_url,
                    "source": "pdf",
                    "pdf_url": pdf_url,
                    "sections": [{
                        "title": "Full Act PDF",
                        "url": pdf_url,
                        "text": pdf_text
                    }]
                }
            else:
                logger.warning("  PDF extraction yielded empty text")
                return {
                    "title": title,
                    "url": act_url,
                    "source": "pdf_empty",
                    "pdf_url": pdf_url,
                    "sections": []
                }
                
        except Exception as pdf_error:
            logger.error(f"  PDF download/extraction failed: {pdf_error}")
            await page.close()
            return {
                "title": title,
                "url": act_url,
                "source": "pdf_failed",
                "pdf_url": pdf_url,
                "sections": []
            }
            
    except Exception as e:
        logger.error(f"Error scraping {act_url}: {e}", exc_info=True)
        if page:
            try:
                await page.close()
            except:
                pass
        return {
            "title": "Unknown Act",
            "url": act_url,
            "source": "error",
            "pdf_url": None,
            "sections": []
        }


async def scrape_acts_for_year(context, year_url: str, year: str):
    """
    Scrape all acts for a given year with resumability.
    
    Args:
        context: Playwright browser context
        year_url: URL of the year page
        year: Year string (e.g., "2023")
        
    Returns:
        List of act data dictionaries
    """
    page = None
    try:
        # Load existing year data for resumability
        year_file = os.path.join(OUTPUT_DIR, f"year_{year}.json")
        existing_data = load_year_data(year_file)
        scraped_urls = get_scraped_act_urls(existing_data)
        
        # Initialize year data if needed
        if existing_data.get("year") is None:
            existing_data = {
                "year": year,
                "url": year_url,
                "acts": []
            }
        
        # Create a page for navigating the year
        page = await context.new_page()
        
        # Navigate to year URL with retry
        async def navigate_to_year():
            await page.goto(year_url, wait_until='domcontentloaded', timeout=10000)
        
        await retry_with_backoff(navigate_to_year)
        
        # Set results per page to maximum if possible
        try:
            await page.select_option('form.mrg.pull-right select[name="rpp"]', '100')
            await page.click('input.btn.btn-success.btn-xs[name="submit_browse"]')
            await page.wait_for_load_state('domcontentloaded')
            await asyncio.sleep(1)
        except:
            pass  # If pagination options don't exist, continue
        
        acts = existing_data.get("acts", []).copy()
        processed_urls = set(scraped_urls)
        
        page_num = 1
        while True:
            # Wait for act links to load
            try:
                await page.wait_for_selector(SELECTORS["act_links"], timeout=10000)
            except:
                logger.info(f"No act links found on page {page_num} for year {year}")
                break
            
            # Get all act links on current page
            act_links = await page.query_selector_all(SELECTORS["act_links"])
            
            if not act_links:
                logger.info(f"No act links found, stopping for year {year}")
                break
            
            logger.info(f"  Page {page_num}: Found {len(act_links)} acts")
            
            for idx, link in enumerate(act_links, 1):
                href = await link.get_attribute("href")
                if href:
                    act_url = urljoin(page.url, href)
                    
                    # Skip if already processed
                    if act_url in processed_urls:
                        logger.info(f"  [{idx}/{len(act_links)}] Skipping (already scraped)")
                        continue
                    
                    processed_urls.add(act_url)
                    
                    logger.info(f"  [{idx}/{len(act_links)}] Scraping: {act_url[:80]}...")
                    
                    # Scrape the act (PDF only)
                    act_data = await scrape_act_pdf(context, act_url)
                    
                    if act_data:
                        acts.append(act_data)
                        source = act_data.get("source", "unknown")
                        sections_count = len(act_data.get("sections", []))
                        logger.info(f"  ✓ Completed: source={source}, sections={sections_count}")
                        
                        # Save progress after each act
                        existing_data["acts"] = acts
                        safe_json_write(year_file, existing_data)
                    
                    # Rate limiting between acts
                    await asyncio.sleep(1.5)
            
            # Check for next page
            next_button = await page.query_selector("div.panel.panel-primary > div > a.pull-right")
            if next_button:
                next_href = await next_button.get_attribute('href')
                if next_href and next_href != '#':
                    logger.info(f"  Moving to next page...")
                    page_num += 1
                    await next_button.click()
                    await page.wait_for_load_state('domcontentloaded')
                    await asyncio.sleep(2)
                else:
                    break
            else:
                break
        
        await page.close()
        
        # Save final year data
        if acts:
            logger.info(f"Year {year} complete: {len(acts)} acts scraped")
            safe_json_write(year_file, {"year": year, "url": year_url, "acts": acts})
        else:
            logger.warning(f"Year {year}: No acts found, saved to temp_years/")
            temp_year_path = os.path.join(TEMP_YEARS_DIR, f"year_{year}.json")
            safe_json_write(temp_year_path, {"year": year, "url": year_url, "acts": []})
        
        return acts
        
    except Exception as e:
        logger.error(f"Error scraping year {year}: {e}", exc_info=True)
        if page:
            try:
                await page.close()
            except:
                pass
        return existing_data.get("acts", [])


async def main():
    """Main function to orchestrate the scraping process."""
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(TEMP_YEARS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Load years_data.json (required)
    years_data_file = 'years_data.json'
    if os.path.exists(years_data_file):
        with open(years_data_file, 'r', encoding='utf-8') as f:
            all_years = json.load(f)
        logger.info(f"Loaded {len(all_years['years'])} years from {years_data_file}")
    else:
        raise FileNotFoundError(f"{years_data_file} not found. Please run Phase 1 scraper first.")
    
    # Filter years: skip those already saved as JSON
    years_to_process = [
        year for year in all_years["years"]
        if not os.path.exists(os.path.join(OUTPUT_DIR, f"year_{year['year']}.json"))
    ]
    
    if not years_to_process:
        logger.info("All years already have JSON files! Nothing to process.")
        return
    
    logger.info(f"Found {len(years_to_process)} years to process (out of {len(all_years['years'])} total)\n")
    
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, slow_mo=300)
        
        try:
            for idx, year_data in enumerate(years_to_process, 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"[{idx}/{len(years_to_process)}] Processing year {year_data['year']}")
                logger.info(f"{'='*60}")
                
                # Create new context for each year (prevents memory leaks)
                context = await browser.new_context(accept_downloads=False)
                
                try:
                    acts = await scrape_acts_for_year(context, year_data["url"], year_data["year"])
                    logger.info(f"Year {year_data['year']} completed: {len(acts)} acts")
                except Exception as e:
                    logger.error(f"Error scraping year {year_data['year']}: {e}", exc_info=True)
                finally:
                    # Clean up context
                    try:
                        # Close all pages first
                        for page in context.pages:
                            try:
                                await page.close()
                            except:
                                pass
                        await context.close()
                        logger.info("Context closed successfully")
                    except Exception as close_err:
                        logger.error(f"Error closing context: {close_err}")
                    
                    # Breathing room between years
                    await asyncio.sleep(3)
                
                # Restart browser every 20 years to prevent memory issues
                if idx % 20 == 0 and idx < len(years_to_process):
                    logger.info("\n>>> Restarting browser to free memory...")
                    try:
                        await browser.close()
                        await asyncio.sleep(5)
                        browser = await pw.chromium.launch(headless=False, slow_mo=300)
                        logger.info(">>> Browser restarted successfully")
                    except Exception as restart_err:
                        logger.error(f"Browser restart failed: {restart_err}")
        
        finally:
            try:
                await browser.close()
                logger.info("Browser closed successfully")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
        
        logger.info("\n" + "="*60)
        logger.info("SCRAPING COMPLETE")
        logger.info("="*60)
        
        # Generate statistics from saved files
        stats = {
            "total_acts": 0,
            "pdf": 0,
            "pdf_empty": 0,
            "pdf_failed": 0,
            "no_pdf": 0,
            "error": 0
        }
        
        for filename in os.listdir(OUTPUT_DIR):
            if filename.startswith("year_") and filename.endswith(".json"):
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    year_data = json.load(f)
                    for act in year_data.get("acts", []):
                        stats["total_acts"] += 1
                        source = act.get("source", "unknown")
                        if source in stats:
                            stats[source] += 1
        
        logger.info(f"\nStatistics:")
        logger.info(f"  Total acts: {stats['total_acts']}")
        logger.info(f"  PDF extracted: {stats['pdf']}")
        logger.info(f"  PDF empty: {stats['pdf_empty']}")
        logger.info(f"  PDF failed: {stats['pdf_failed']}")
        logger.info(f"  No PDF found: {stats['no_pdf']}")
        logger.info(f"  Errors: {stats['error']}")
        logger.info(f"\nData saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠ Interrupted by user. Progress saved safely.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)