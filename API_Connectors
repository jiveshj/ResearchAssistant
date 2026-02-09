"""
Modular API Connectors for Research Paper Sources
Supports ArXiv and PubMed with extensible architecture
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Optional
from research_assistant import Paper
import time
from abc import ABC, abstractmethod


class PaperConnector(ABC):
    """
    Abstract base class for paper source connectors
    Enables easy addition of new sources (e.g., Semantic Scholar, Google Scholar)
    """
    
    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search for papers matching query"""
        pass
    
    @abstractmethod
    def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """Retrieve specific paper by ID"""
        pass


class ArXivConnector(PaperConnector):
    """
    Connector for ArXiv API
    
    ArXiv covers: Computer Science, Physics, Mathematics, etc.
    API Docs: https://arxiv.org/help/api/user-manual
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, rate_limit_delay: float = 3.0):
        """
        Args:
            rate_limit_delay: Seconds to wait between API calls (ArXiv requires 3s)
        """
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """
        Search ArXiv for papers
        
        Args:
            query: Search query (e.g., "machine learning transformers")
            max_results: Maximum number of results to return
            
        Returns:
            List of Paper objects
        """
        print(f"🔍 Searching ArXiv for: '{query}'...")
        
        self._rate_limit()
        
        # Construct API request
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse XML response
            papers = self._parse_arxiv_response(response.text)
            
            print(f"✅ Found {len(papers)} papers from ArXiv\n")
            return papers
            
        except requests.RequestException as e:
            print(f"❌ ArXiv API error: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_text: str) -> List[Paper]:
        """Parse ArXiv XML response into Paper objects"""
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
            
            # ArXiv uses Atom namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                # Extract paper metadata
                title = entry.find('atom:title', ns).text.strip()
                
                # Get authors
                authors = [
                    author.find('atom:name', ns).text
                    for author in entry.findall('atom:author', ns)
                ]
                
                # Get abstract (summary in ArXiv XML)
                abstract = entry.find('atom:summary', ns).text.strip()
                
                # Get paper ID from URL
                paper_url = entry.find('atom:id', ns).text
                paper_id = paper_url.split('/abs/')[-1]
                
                papers.append(Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=paper_url,
                    source='arxiv',
                    paper_id=paper_id
                ))
        
        except ET.ParseError as e:
            print(f"❌ Error parsing ArXiv XML: {e}")
        
        return papers
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """
        Get specific ArXiv paper by ID
        
        Args:
            paper_id: ArXiv ID (e.g., "1706.03762")
        """
        self._rate_limit()
        
        params = {
            'id_list': paper_id,
            'max_results': 1
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            papers = self._parse_arxiv_response(response.text)
            return papers[0] if papers else None
        
        except requests.RequestException as e:
            print(f"❌ Error fetching ArXiv paper {paper_id}: {e}")
            return None


class PubMedConnector(PaperConnector):
    """
    Connector for PubMed API
    
    PubMed covers: Biomedical and life sciences literature
    API Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
    """
    
    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    
    def __init__(self, email: Optional[str] = None, rate_limit_delay: float = 0.34):
        """
        Args:
            email: Your email (recommended by NCBI for API key)
            rate_limit_delay: Delay between requests (3 requests/second limit)
        """
        self.email = email
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """
        Search PubMed for papers
        
        Args:
            query: Search query (e.g., "CRISPR gene editing")
            max_results: Maximum results to return
            
        Returns:
            List of Paper objects
        """
        print(f"🔍 Searching PubMed for: '{query}'...")
        
        # Step 1: Search for PMIDs
        pmids = self._search_pmids(query, max_results)
        
        if not pmids:
            print("⚠️ No results found in PubMed\n")
            return []
        
        # Step 2: Fetch paper details for each PMID
        papers = self._fetch_papers(pmids)
        
        print(f"✅ Found {len(papers)} papers from PubMed\n")
        return papers
    
    def _search_pmids(self, query: str, max_results: int) -> List[str]:
        """Search PubMed and get list of PMIDs (PubMed IDs)"""
        self._rate_limit()
        
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json'
        }
        
        if self.email:
            params['email'] = self.email
        
        try:
            response = requests.get(self.SEARCH_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            pmids = data.get('esearchresult', {}).get('idlist', [])
            return pmids
        
        except requests.RequestException as e:
            print(f"❌ PubMed search error: {e}")
            return []
    
    def _fetch_papers(self, pmids: List[str]) -> List[Paper]:
        """Fetch full paper details for list of PMIDs"""
        papers = []
        
        for pmid in pmids:
            paper = self.get_paper_by_id(pmid)
            if paper:
                papers.append(paper)
        
        return papers
    
    def get_paper_by_id(self, pmid: str) -> Optional[Paper]:
        """
        Fetch specific PubMed paper by PMID
        
        Args:
            pmid: PubMed ID
        """
        self._rate_limit()
        
        params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml'
        }
        
        if self.email:
            params['email'] = self.email
        
        try:
            response = requests.get(self.FETCH_URL, params=params, timeout=10)
            response.raise_for_status()
            
            paper = self._parse_pubmed_xml(response.text, pmid)
            return paper
        
        except requests.RequestException as e:
            print(f"❌ Error fetching PubMed paper {pmid}: {e}")
            return None
    
    def _parse_pubmed_xml(self, xml_text: str, pmid: str) -> Optional[Paper]:
        """Parse PubMed XML response"""
        try:
            root = ET.fromstring(xml_text)
            article = root.find('.//PubmedArticle')
            
            if article is None:
                return None
            
            # Extract title
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "No title"
            
            # Extract authors
            author_list = article.findall('.//Author')
            authors = []
            for author in author_list:
                lastname = author.find('LastName')
                forename = author.find('ForeName')
                if lastname is not None and forename is not None:
                    authors.append(f"{forename.text} {lastname.text}")
            
            # Extract abstract
            abstract_elem = article.find('.//AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
            
            # Construct URL
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            
            return Paper(
                title=title,
                authors=authors if authors else ["Unknown"],
                abstract=abstract,
                url=url,
                source='pubmed',
                paper_id=pmid
            )
        
        except ET.ParseError as e:
            print(f"❌ Error parsing PubMed XML: {e}")
            return None


class MultiSourceConnector:
    """
    Aggregates multiple paper sources
    Enables searching across ArXiv + PubMed simultaneously
    """
    
    def __init__(self, connectors: List[PaperConnector]):
        """
        Args:
            connectors: List of connector instances to use
        """
        self.connectors = connectors
    
    def search_all(self, query: str, max_results_per_source: int = 10) -> List[Paper]:
        """
        Search across all connected sources
        
        Args:
            query: Search query
            max_results_per_source: Max results from each source
            
        Returns:
            Combined list of papers from all sources
        """
        all_papers = []
        
        for connector in self.connectors:
            papers = connector.search(query, max_results_per_source)
            all_papers.extend(papers)
        
        # Deduplicate by title (simple approach)
        seen_titles = set()
        unique_papers = []
        
        for paper in all_papers:
            if paper.title not in seen_titles:
                seen_titles.add(paper.title)
                unique_papers.append(paper)
        
        return unique_papers


# Example usage
if __name__ == "__main__":
    # Initialize connectors
    arxiv = ArXivConnector()
    pubmed = PubMedConnector(email="your.email@example.com")  # Optional email
    
    # Test ArXiv
    print("="*80)
    print("TESTING ARXIV CONNECTOR")
    print("="*80)
    arxiv_papers = arxiv.search("transformers attention mechanism", max_results=3)
    for paper in arxiv_papers:
        print(f"📄 {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        print(f"   URL: {paper.url}\n")
    
    # Test PubMed
    print("="*80)
    print("TESTING PUBMED CONNECTOR")
    print("="*80)
    pubmed_papers = pubmed.search("CRISPR gene editing", max_results=3)
    for paper in pubmed_papers:
        print(f"📄 {paper.title}")
        print(f"   Authors: {', '.join(paper.authors[:3])}")
        print(f"   URL: {paper.url}\n")
    
    # Test MultiSource
    print("="*80)
    print("TESTING MULTI-SOURCE SEARCH")
    print("="*80)
    multi_connector = MultiSourceConnector([arxiv, pubmed])
    all_papers = multi_connector.search_all("deep learning", max_results_per_source=5)
    print(f"Total papers found: {len(all_papers)}")
    print(f"ArXiv: {sum(1 for p in all_papers if p.source == 'arxiv')}")
    print(f"PubMed: {sum(1 for p in all_papers if p.source == 'pubmed')}")
