import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_distribution = {}
    links = corpus[page]
    total_pages = len(corpus)

    # If the current page has no outgoing links, treat it as if it links to all pages
    if not links:
        for p in corpus:
            probability_distribution[p] = 1 / total_pages
        return probability_distribution

    # With damping factor, we can either choose one of the links, or go to any page
    for p in corpus:
        probability_distribution[p] = (1 - damping_factor) / total_pages

    for link in links:
        probability_distribution[link] += damping_factor / len(links)

    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = {page: 0 for page in corpus}
    page = random.choice(list(corpus.keys()))  # Start with a random page
    pagerank[page] += 1

    for _ in range(1, n):
        model = transition_model(corpus, page, damping_factor)
        page = random.choices(list(model.keys()), weights=list(model.values()), k=1)[0]
        pagerank[page] += 1

    # Normalize the values to sum to 1
    pagerank = {page: count / n for page, count in pagerank.items()}
    
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    pagerank = {page: 1 / N for page in corpus}
    new_rank = {page: 0 for page in corpus}

    converged = False
    while not converged:
        converged = True
        for page in corpus:
            total = 0
            for other_page in corpus:
                if page in corpus[other_page] or not corpus[other_page]:
                    total += pagerank[other_page] / (len(corpus[other_page]) if corpus[other_page] else N)

            new_rank[page] = (1 - damping_factor) / N + damping_factor * total

        for page in pagerank:
            if abs(new_rank[page] - pagerank[page]) > 0.001:
                converged = False

        pagerank = new_rank.copy()

    return pagerank


if __name__ == "__main__":
    main()
