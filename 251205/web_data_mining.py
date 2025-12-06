# import
import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
from urllib.parse import urljoin


# functions
def get_links(url):
    """
    Fetches all hyperlinks from the given URL.

    Args:
        url (str): The URL to fetch links from.

    Returns:
        list: A list of hyperlinks found on the page.
    """

    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("http://") or href.startswith("https://"):
            links.append(urljoin(url, href))

    return links


def build_graph(seed_url, max_depth=1):
    """Builds a directed graph of hyperlinks up to a specified depth."""
    graph = nx.DiGraph()
    visited = set()

    def crawl(url, depth):
        if depth > max_depth or url in visited:
            return
        visited.add(url)
        links = get_links(url)
        for link in links:
            graph.add_edge(url, link)
            crawl(link, depth + 1)

    crawl(seed_url, 0)

    return graph


def draw_graph(graph, filename="graph.png"):
    """Draws the graph and saves it to a file."""
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=50,
        node_color="lightblue",
        font_size=8,
        font_color="black",
        edge_color="gray",
    )
    plt.title("Web Structure Graph")
    plt.savefig(filename)
    plt.show()
    plt.close()


def main():
    seed_url = "https://webdriveruniversity.com/"
    graph = build_graph(seed_url, max_depth=1)
    draw_graph(graph, "web_structure_graph.png")


if __name__ == "__main__":
    main()