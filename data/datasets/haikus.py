#!/usr/bin/env python3

"""A test script to download haikus from individual websites."""
import pickle
from collections import deque
from multiprocessing import Pool
from pathlib import Path

from requests_html import HTMLSession


def henderson():
    session = HTMLSession()

    url = "http://www.hsa-haiku.org/hendersonawards/henderson.htm"
    r = session.get(url)
    haikus = r.html.find("td > blockquote > p")
    haikus = [h.text for h in haikus]

    print(url, "->", len(haikus))
    return {url: haikus}


def brady():
    session = HTMLSession()

    url = "http://www.hsa-haiku.org/bradyawards/brady.htm"
    r = session.get(url)
    haikus = r.html.find("td > blockquote > p")
    haikus = [h.text for h in haikus]

    print(url, "->", len(haikus))
    return {url: haikus}


def museum():
    session = HTMLSession()
    url = "http://www.hsa-haiku.org/museumhaikuliteratureawards/museumhaikuliterature-award.htm"
    r = session.get(url)
    # Ignore the haikus in <td></td>s because it'd be too hard to parse out the author names et al.
    haikus = r.html.find("p.haiku")
    haikus = [h.text for h in haikus]

    print(url, "->", len(haikus))
    return {url: haikus}


def virgilio():
    session = HTMLSession()
    url = "http://www.hsa-haiku.org/virgilioawards/virgilio.htm"
    r = session.get(url)
    # Not just <p></p>s...
    haikus = r.html.find(".haiku")
    haikus = [h.text for h in haikus]

    print(url, "->", len(haikus))
    return {url: haikus}


def perdiem():
    session = HTMLSession()
    # A massive agglomeration of individual archives, all with the same formatting :D
    url = "https://www.thehaikufoundation.org/per-diem-archive/"
    r = session.get(url)
    urls = r.html.find("li > a")
    urls = (u.attrs["href"] for u in urls)
    urls = filter(lambda x: "IDcat" in x, urls)
    urls = (f"https://www.thehaikufoundation.org{u}" for u in urls)

    all_haikus = dict()
    for url in urls:
        r = session.get(url)
        # Use a more forgiving encoding because invalid utf-8 characters suck.
        r.html.encoding = "ISO-8859-1"
        haikus = r.html.find("td > pre")
        print(url, "->", len(haikus))
        all_haikus[url] = [h.text for h in haikus]

    return all_haikus


def ahapoetry():
    session = HTMLSession()
    url = "https://www.ahapoetry.com/aadoh/h_dictionary.htm"
    r = session.get(url)
    urls = r.html.find("p > a")
    urls = (u.attrs["href"] for u in urls)
    urls = (f"https://www.ahapoetry.com/aadoh/{u}" for u in urls)

    def is_haiku(x):
        try:
            return x.attrs["align"] == "center"
        except KeyError:
            return False

    all_haikus = dict()
    for url in urls:
        r = session.get(url)
        haikus = r.html.find("p")
        haikus = filter(is_haiku, haikus)
        all_haikus[url] = [h.text for h in haikus]
        print(url, "->", len(all_haikus[url]))

    return all_haikus


def dailyhaiku():
    session = HTMLSession()
    baseurl = "http://www.dailyhaiku.org/haiku/"
    # Page numbers go from 1..519, determined experimentally.
    urls = (f"{baseurl}?pg={i}" for i in range(1, 520))
    all_haikus = dict()
    for url in urls:
        r = session.get(url)
        haikus = r.html.find("p.haiku")
        all_haikus[url] = [h.text for h in haikus]
        print(url, "->", len(all_haikus[url]))

    return all_haikus


def tinywords():
    session = HTMLSession()
    url = "http://tinywords.com/haiku/?sort=date&order=1&show=all"
    r = session.get(url)
    urls = r.html.find("td.nowrap > a")
    urls = (u.attrs["href"] for u in urls)
    urls = (f"http://tinywords.com{u}" for u in urls)

    def remove_author(s):
        head, _, _ = s.rpartition("—")
        return head

    all_haikus = []
    for url in urls:
        r = session.get(url)
        haikus = r.html.find("pre")
        # TODO: This removes newlines in the <pre> tag.
        haikus = [remove_author(h.text) for h in haikus]
        all_haikus += haikus

    print("http://tinywords.com/haiku/", "->", len(all_haikus))
    # Unfortunately each haiku is on its own page, so index them under one URL.
    return {"http://tinywords.com/haiku/": all_haikus}


def heronsnest():
    """Recursively scrape The Herons Nest for haikus.

    Visit any unvisited absolute link to a page on www.theheronsnest.com domain, and call any
        <p class="haiku"></p>
        <p><font></font></p>
    a haiku.
    """
    all_haikus = {}
    session = HTMLSession()
    baseurl = "http://www.theheronsnest.com/archives.html"
    q = deque()
    q.append(baseurl)
    visited = {
        baseurl,
        # Manually mark some URLs as visited to avoid visiting them.
        "http://www.theheronsnest.com/index.html",
        "http://www.theheronsnest.com/awards/index.html",
        "http://www.theheronsnest.com/staff.html",
        "http://www.theheronsnest.com/submit.html",
        "http://www.theheronsnest.com/friends.html",
        "http://www.theheronsnest.com/memorials/",
        "http://www.theheronsnest.com/order.html",
        "http://www.thewondercode.com/#payment",
        "http://www.theheronsnest.com/archived_issues/connections/",
        "http://www.theheronsnest.com/archived_issues/journal/",
        "http://www.theheronsnest.com/archived_issues/journal",
        "http://www.theheronsnest.com/archived_issues/haiku/",
        "http://www.theheronsnest.com/archived_issues/haiku/index.html",
    }

    while q:
        url = q.pop()
        r = session.get(url)
        # Be very forgiving with unicode.
        r.html.encoding = "ISO-8859-1"
        for link in r.html.absolute_links:
            if "#" in link:
                link, _, _ = link.partition("#")
            # Don't leave the Herons Nest website, or revisit a link!
            if link not in visited and link.startswith("http://www.theheronsnest.com/"):
                q.append(link)
                visited.add(link)

        if not ("thn_va" in link or "thn_toc" in link):
            haikus = [h.text for h in r.html.find("p.haiku")]
            # TODO: This has a lot of false positives.
            haikus += [h.text for h in r.html.find("p > font")]
            # Pages with fewer than 4 haikus tend to be false positives
            if len(haikus) > 4:
                print(url, "->", len(haikus))
                all_haikus[url] = haikus

    return all_haikus


def download(job):
    return job()


if __name__ == "__main__":
    jobs = [
        henderson,
        brady,
        museum,
        virgilio,
        perdiem,
        ahapoetry,
        dailyhaiku,
        tinywords,
        heronsnest,
    ]

    # Default to nprocs
    with Pool(processes=None) as pool:
        results = pool.map(download, jobs)

    # Glue together list of dictionaries into a single dictionary.
    haikus = dict()
    for result in results:
        haikus.update(result)

    s = 0
    for _, values in haikus.items():
        s += len(values)

    print("Saving", s, "haikus to haikus.pkl")

    with open(Path(__file__).parent.parent.joinpath("haikus.pkl"), "wb") as f:
        pickle.dump(haikus, f)
