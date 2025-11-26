#!/usr/bin/env python3
"""
Colnect Stamp Scraper - Educational Purpose Only

IMPORTANT: This script is for educational purposes. Always:
- Check and comply with website Terms of Service
- Use rate limiting to avoid overloading servers
- Consider using official APIs when available
- Respect robots.txt
- For production use, obtain proper permissions

This script demonstrates web scraping techniques for collecting
stamp variant data for training ML models.
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from urllib.parse import urljoin, urlparse
import hashlib


class ColnectScraper:
    """
    Scraper for Colnect stamp database.
    Focuses on documented variants and errors.
    """
    
    def __init__(self, output_dir: str = 'data/scraped_stamps',
                 rate_limit: float = 2.0):
        """
        Args:
            output_dir: Where to save scraped data
            rate_limit: Seconds to wait between requests
        """
        self.base_url = 'https://colnect.com/en/stamps'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Research Bot)',
            'Accept': 'text/html,application/xhtml+xml',
        })
        
        self.metadata = []
        
    def rate_limit_wait(self):
        """Respect rate limiting."""
        time.sleep(self.rate_limit)
    
    def download_image(self, url: str, filename: str) -> bool:
        """
        Download an image from URL.
        
        Args:
            url: Image URL
            filename: Local filename to save as
        
        Returns:
            True if successful
        """
        try:
            self.rate_limit_wait()
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                filepath = self.output_dir / filename
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                print(f"Failed to download {url}: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def search_variants(self, search_term: str = 'error',
                       max_pages: int = 5) -> List[Dict]:
        """
        Search for stamps with variants/errors.
        
        Args:
            search_term: Search query (e.g., 'error', 'variety', 'misprint')
            max_pages: Maximum number of result pages to scrape
        
        Returns:
            List of stamp metadata dictionaries
        """
        results = []
        
        print(f"Searching for: {search_term}")
        
        for page in range(1, max_pages + 1):
            print(f"Processing page {page}...")
            
            search_url = f"{self.base_url}/list/search/{search_term}/page/{page}"
            
            try:
                self.rate_limit_wait()
                response = self.session.get(search_url, timeout=30)
                
                if response.status_code != 200:
                    print(f"Failed to fetch page {page}")
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find stamp listings (structure may vary)
                # This is a simplified example - actual implementation needs
                # to be adapted to current Colnect structure
                stamp_items = soup.find_all('div', class_='item_box')
                
                if not stamp_items:
                    print("No more results found")
                    break
                
                for item in stamp_items:
                    stamp_data = self.parse_stamp_item(item)
                    if stamp_data:
                        results.append(stamp_data)
                
            except Exception as e:
                print(f"Error processing page {page}: {e}")
                continue
        
        return results
    
    def parse_stamp_item(self, item) -> Optional[Dict]:
        """
        Parse a stamp item from search results.
        
        Args:
            item: BeautifulSoup element
        
        Returns:
            Dictionary with stamp metadata
        """
        try:
            # Extract stamp information
            # NOTE: These selectors are examples and need to be updated
            # based on actual Colnect HTML structure
            
            title_elem = item.find('a', class_='item_link')
            if not title_elem:
                return None
            
            title = title_elem.text.strip()
            detail_url = urljoin(self.base_url, title_elem['href'])
            
            # Extract image URL
            img_elem = item.find('img')
            image_url = img_elem['src'] if img_elem else None
            
            # Create unique ID from URL
            stamp_id = hashlib.md5(detail_url.encode()).hexdigest()[:12]
            
            return {
                'id': stamp_id,
                'title': title,
                'detail_url': detail_url,
                'image_url': image_url,
                'source': 'colnect',
                'search_term': 'error'  # Add context
            }
            
        except Exception as e:
            print(f"Error parsing item: {e}")
            return None
    
    def scrape_stamp_details(self, stamp_data: Dict) -> Dict:
        """
        Scrape detailed information for a specific stamp.
        
        Args:
            stamp_data: Basic stamp metadata
        
        Returns:
            Enhanced metadata with details
        """
        try:
            self.rate_limit_wait()
            response = self.session.get(stamp_data['detail_url'], timeout=30)
            
            if response.status_code != 200:
                return stamp_data
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract detailed information
            # NOTE: Update selectors based on actual structure
            
            # Description
            desc_elem = soup.find('div', class_='description')
            if desc_elem:
                stamp_data['description'] = desc_elem.text.strip()
            
            # Year
            year_elem = soup.find('span', class_='year')
            if year_elem:
                stamp_data['year'] = year_elem.text.strip()
            
            # Country
            country_elem = soup.find('span', class_='country')
            if country_elem:
                stamp_data['country'] = country_elem.text.strip()
            
            # Catalog numbers
            catalog_elem = soup.find('div', class_='catalog_numbers')
            if catalog_elem:
                stamp_data['catalog_numbers'] = catalog_elem.text.strip()
            
            # Look for variant information
            variant_info = self.extract_variant_info(soup)
            if variant_info:
                stamp_data['variant'] = variant_info
            
            # Find reference image (normal version)
            related_stamps = soup.find_all('div', class_='related_stamp')
            if related_stamps:
                stamp_data['related_stamps'] = []
                for related in related_stamps[:3]:  # Limit to 3 related
                    img = related.find('img')
                    if img:
                        stamp_data['related_stamps'].append({
                            'image_url': img['src'],
                            'caption': related.text.strip()
                        })
            
            return stamp_data
            
        except Exception as e:
            print(f"Error scraping details for {stamp_data['id']}: {e}")
            return stamp_data
    
    def extract_variant_info(self, soup) -> Optional[Dict]:
        """
        Extract information about the variant/error.
        
        Args:
            soup: BeautifulSoup object
        
        Returns:
            Dictionary with variant details
        """
        variant_info = {}
        
        # Look for keywords indicating variant type
        text_content = soup.get_text().lower()
        
        if 'inverted' in text_content or 'invert' in text_content:
            variant_info['type'] = 'inverted'
        elif 'missing color' in text_content or 'color error' in text_content:
            variant_info['type'] = 'color_error'
        elif 'double' in text_content and 'print' in text_content:
            variant_info['type'] = 'double_print'
        elif 'imperforate' in text_content:
            variant_info['type'] = 'perforation_error'
        elif 'watermark' in text_content:
            variant_info['type'] = 'watermark_error'
        
        # Extract specific difference description
        desc_elem = soup.find('div', {'class': 'variant_description'})
        if desc_elem:
            variant_info['description'] = desc_elem.text.strip()
        
        return variant_info if variant_info else None
    
    def download_stamp_images(self, stamp_data: Dict) -> Dict:
        """
        Download images for a stamp (variant and reference if available).
        
        Args:
            stamp_data: Stamp metadata
        
        Returns:
            Updated metadata with local file paths
        """
        stamp_id = stamp_data['id']
        
        # Download main image (variant)
        if stamp_data.get('image_url'):
            filename = f"{stamp_id}_variant.jpg"
            if self.download_image(stamp_data['image_url'], filename):
                stamp_data['local_image'] = str(self.output_dir / filename)
        
        # Download related stamps (potential reference images)
        if stamp_data.get('related_stamps'):
            stamp_data['related_local_images'] = []
            for idx, related in enumerate(stamp_data['related_stamps']):
                filename = f"{stamp_id}_related_{idx}.jpg"
                if self.download_image(related['image_url'], filename):
                    stamp_data['related_local_images'].append({
                        'path': str(self.output_dir / filename),
                        'caption': related['caption']
                    })
        
        return stamp_data
    
    def scrape_dataset(self, search_terms: List[str],
                      max_stamps_per_term: int = 50) -> List[Dict]:
        """
        Scrape a complete dataset of stamp variants.
        
        Args:
            search_terms: List of search queries
            max_stamps_per_term: Maximum stamps to scrape per search term
        
        Returns:
            List of stamp metadata
        """
        all_stamps = []
        
        for term in search_terms:
            print(f"\n=== Searching for: {term} ===")
            
            # Search for stamps
            results = self.search_variants(term, max_pages=10)
            
            # Limit results
            results = results[:max_stamps_per_term]
            
            print(f"Found {len(results)} stamps for '{term}'")
            
            # Scrape details and download images
            for idx, stamp_data in enumerate(results, 1):
                print(f"Processing {idx}/{len(results)}: {stamp_data['title']}")
                
                # Get detailed info
                stamp_data = self.scrape_stamp_details(stamp_data)
                
                # Download images
                stamp_data = self.download_stamp_images(stamp_data)
                
                all_stamps.append(stamp_data)
                
                # Save progress periodically
                if idx % 10 == 0:
                    self.save_metadata(all_stamps)
        
        # Final save
        self.save_metadata(all_stamps)
        
        return all_stamps
    
    def save_metadata(self, stamps: List[Dict]):
        """Save metadata to JSON file."""
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(stamps, f, indent=2)
        print(f"Saved metadata for {len(stamps)} stamps to {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Scrape stamp variants from Colnect (Educational Purpose)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT LEGAL AND ETHICAL NOTICE:
This script is for EDUCATIONAL PURPOSES ONLY.

Before using this script, you MUST:
1. Read and comply with Colnect's Terms of Service
2. Check and respect robots.txt
3. Use appropriate rate limiting (default: 2 seconds)
4. Consider using official APIs when available
5. Obtain proper permissions for production use
6. Ensure your use case is legal in your jurisdiction

The authors are not responsible for misuse of this script.
        """
    )
    parser.add_argument('--output', type=str, default='data/scraped_stamps',
                       help='Output directory')
    parser.add_argument('--max-per-term', type=int, default=50,
                       help='Maximum stamps per search term')
    parser.add_argument('--rate-limit', type=float, default=2.0,
                       help='Seconds between requests (minimum: 1.0)')
    
    args = parser.parse_args()
    
    # Enforce minimum rate limit
    if args.rate_limit < 1.0:
        print("Warning: Rate limit must be at least 1.0 seconds. Setting to 1.0.")
        args.rate_limit = 1.0
    
    # Important notice
    print("=" * 70)
    print("IMPORTANT: Web Scraping Ethics and Legality")
    print("=" * 70)
    print("Before proceeding, ensure you:")
    print("1. Have read and comply with Colnect's Terms of Service")
    print("2. Are using this for educational/research purposes")
    print("3. Will respect rate limiting and server resources")
    print("4. Consider using official APIs when available")
    print("5. Have proper permissions for your use case")
    print("=" * 70)
    print("\nNOTE: This script uses example HTML selectors that may not work")
    print("with the current Colnect website structure. You will need to")
    print("inspect the website and update the selectors accordingly.")
    print("=" * 70)
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    print()
    
    # Create scraper
    scraper = ColnectScraper(
        output_dir=args.output,
        rate_limit=args.rate_limit
    )
    
    # Define search terms for variants
    search_terms = [
        'error',
        'variety',
        'misprint',
        'inverted',
        'missing color',
        'double print',
        'perforation error',
        'watermark error'
    ]
    
    # Scrape dataset
    stamps = scraper.scrape_dataset(
        search_terms,
        max_stamps_per_term=args.max_per_term
    )
    
    print(f"\n{'='*70}")
    print(f"Scraping complete!")
    print(f"Total stamps scraped: {len(stamps)}")
    print(f"Images saved to: {args.output}")
    print(f"Metadata saved to: {args.output}/metadata.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
