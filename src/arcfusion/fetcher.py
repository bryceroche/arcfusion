"""
arXiv Paper Fetcher - Retrieve ML papers for component extraction.

Fetches papers from arXiv API, extracts abstracts, and feeds them
to the decomposer pipeline.
"""

import arxiv
from dataclasses import dataclass
from typing import Iterator, Optional
from .db import ArcFusionDB, ProcessedPaper
from .decomposer import PaperDecomposer


@dataclass
class ArxivPaper:
    """Represents a fetched arXiv paper."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    pdf_url: str
    published: str
    categories: list[str]


class ArxivFetcher:
    """Fetch papers from arXiv API."""

    # Categories relevant to ML architectures
    ML_CATEGORIES = [
        "cs.LG",   # Machine Learning
        "cs.CL",   # Computation and Language
        "cs.CV",   # Computer Vision
        "cs.AI",   # Artificial Intelligence
        "cs.NE",   # Neural and Evolutionary Computing
        "stat.ML", # Statistics - Machine Learning
    ]

    # Search terms for architecture papers
    ARCHITECTURE_TERMS = [
        "transformer",
        "attention mechanism",
        "state space model",
        "mamba",
        "neural architecture",
        "language model",
        "self-attention",
        "mixture of experts",
        "linear attention",
        "recurrent neural",
        "convolutional neural",
        "graph neural",
        "vision transformer",
        "diffusion model",
    ]

    def __init__(self, db: ArcFusionDB):
        self.db = db
        self.decomposer = PaperDecomposer(db)
        self.client = arxiv.Client()

    def _normalize_arxiv_id(self, entry_id: str) -> str:
        """Extract arxiv_id from entry URL."""
        # entry_id is like "http://arxiv.org/abs/2312.00752v1"
        arxiv_id = entry_id.split("/")[-1]
        # Remove version
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.rsplit("v", 1)[0]
        return arxiv_id

    def _result_to_paper(self, result: arxiv.Result) -> ArxivPaper:
        """Convert arxiv API result to ArxivPaper dataclass."""
        return ArxivPaper(
            arxiv_id=self._normalize_arxiv_id(result.entry_id),
            title=result.title,
            abstract=result.summary,
            authors=[a.name for a in result.authors],
            pdf_url=result.pdf_url,
            published=result.published.isoformat(),
            categories=result.categories,
        )

    def search(
        self,
        query: str,
        max_results: int = 50,
        sort_by: str = "submitted"
    ) -> Iterator[ArxivPaper]:
        """
        Search arXiv for papers matching query.

        Args:
            query: Search query (supports arXiv query syntax)
            max_results: Maximum number of papers to return
            sort_by: "submitted" (newest), "relevance", or "updated"
        """
        sort_criterion = {
            "submitted": arxiv.SortCriterion.SubmittedDate,
            "relevance": arxiv.SortCriterion.Relevance,
            "updated": arxiv.SortCriterion.LastUpdatedDate,
        }.get(sort_by, arxiv.SortCriterion.SubmittedDate)

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending,
        )

        for result in self.client.results(search):
            yield self._result_to_paper(result)

    def search_architectures(
        self,
        max_results: int = 50,
        category: str = "cs.LG"
    ) -> Iterator[ArxivPaper]:
        """
        Search for ML architecture papers specifically.

        Builds a query targeting papers about neural architectures.
        """
        # Combine terms with OR
        terms = " OR ".join(f'"{term}"' for term in self.ARCHITECTURE_TERMS)
        query = f"cat:{category} AND ({terms})"

        yield from self.search(query, max_results=max_results, sort_by="submitted")

    def fetch_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """Fetch a specific paper by arXiv ID."""
        search = arxiv.Search(id_list=[arxiv_id])

        for result in self.client.results(search):
            return self._result_to_paper(result)
        return None

    def fetch_by_ids(self, arxiv_ids: list[str]) -> Iterator[ArxivPaper]:
        """Fetch multiple papers by arXiv ID."""
        search = arxiv.Search(id_list=arxiv_ids)

        for result in self.client.results(search):
            yield self._result_to_paper(result)

    def ingest_paper(self, paper: ArxivPaper, verbose: bool = True) -> dict:
        """
        Process a single paper through the decomposition pipeline.

        Returns dict with status and details.
        """
        # Check if already processed
        if self.db.is_paper_processed(paper.arxiv_id):
            if verbose:
                print(f"  [SKIP] {paper.arxiv_id}: Already processed")
            return {"status": "skipped", "reason": "already_processed"}

        # Decompose the paper
        try:
            engine, new_components = self.decomposer.create_engine_from_paper(
                title=paper.title,
                abstract=paper.abstract,
                paper_url=paper.pdf_url,
            )

            # Record as processed
            processed = ProcessedPaper(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                engine_id=engine.engine_id,
                status="processed",
                notes=f"Extracted {len(engine.component_ids)} components"
            )
            self.db.add_processed_paper(processed)

            if verbose:
                print(f"  [OK] {paper.arxiv_id}: {paper.title[:60]}...")
                print(f"       -> Engine: {engine.name[:50]}... ({len(engine.component_ids)} components)")
                if new_components:
                    print(f"       -> New components: {[c.name for c in new_components]}")

            return {
                "status": "processed",
                "engine_id": engine.engine_id,
                "component_count": len(engine.component_ids),
                "new_components": len(new_components),
            }

        except Exception as e:
            # Record failure
            processed = ProcessedPaper(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                status="failed",
                notes=str(e)
            )
            self.db.add_processed_paper(processed)

            if verbose:
                print(f"  [FAIL] {paper.arxiv_id}: {e}")

            return {"status": "failed", "error": str(e)}

    def ingest_batch(
        self,
        papers: Iterator[ArxivPaper],
        max_papers: int = 50,
        verbose: bool = True
    ) -> dict:
        """
        Process a batch of papers.

        Returns summary statistics.
        """
        stats = {
            "total": 0,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "new_components": 0,
        }

        if verbose:
            print(f"\n--- Ingesting papers (max {max_papers}) ---\n")

        for i, paper in enumerate(papers):
            if i >= max_papers:
                break

            stats["total"] += 1
            result = self.ingest_paper(paper, verbose=verbose)

            if result["status"] == "processed":
                stats["processed"] += 1
                stats["new_components"] += result.get("new_components", 0)
            elif result["status"] == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1

        if verbose:
            print("\n--- Batch Complete ---")
            print(f"Total: {stats['total']}")
            print(f"Processed: {stats['processed']}")
            print(f"Skipped: {stats['skipped']}")
            print(f"Failed: {stats['failed']}")
            print(f"New components: {stats['new_components']}")

        return stats

    def ingest_search(
        self,
        query: str,
        max_results: int = 50,
        verbose: bool = True
    ) -> dict:
        """Search and ingest papers in one call."""
        papers = self.search(query, max_results=max_results)
        return self.ingest_batch(papers, max_papers=max_results, verbose=verbose)

    def ingest_architectures(
        self,
        max_results: int = 50,
        category: str = "cs.LG",
        verbose: bool = True
    ) -> dict:
        """Search for and ingest ML architecture papers."""
        if verbose:
            print(f"Searching {category} for architecture papers...")
        papers = self.search_architectures(max_results=max_results, category=category)
        return self.ingest_batch(papers, max_papers=max_results, verbose=verbose)
